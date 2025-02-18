# noqa: D100

import functools
import logging
import operator
from typing import Callable, Dict, List, Optional, Tuple, Union

import hail as hl

import gnomad.utils.annotations as annotate_utils
from gnomad.resources.resource_utils import DataException
from gnomad.utils.intervals import pad_intervals
from gnomad.utils.parse import parse_locus_intervals
from gnomad.utils.reference_genome import get_reference_genome

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_to_adj(mt: hl.MatrixTable) -> hl.MatrixTable:
    """Filter genotypes to adj criteria."""
    if "adj" not in list(mt.entry):
        mt = annotate_utils.annotate_adj(mt)
    mt = mt.filter_entries(mt.adj)
    return mt.drop(mt.adj)


def filter_by_frequency(
    t: Union[hl.MatrixTable, hl.Table],
    direction: str,
    frequency: float = None,
    allele_count: int = None,
    population: str = None,
    subpop: str = None,
    downsampling: int = None,
    keep: bool = True,
    adj: bool = True,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter MatrixTable or Table with gnomAD-format frequency data (assumed bi-allelic/split).

    gnomAD frequency data format expectation is: Array[Struct(Array[AC], Array[AF], AN, homozygote_count, meta)].

    At least one of frequency or allele_count is required.

    Subpop can be specified without a population if desired.

    :param t: Input MatrixTable or Table
    :param direction: One of "above", "below", and "equal" (how to apply the filter)
    :param frequency: Frequency to filter by (one of frequency or allele_count is required)
    :param allele_count: Allele count to filter by (one of frequency or allele_count is required)
    :param population: Population in which to filter frequency
    :param subpop: Sub-population in which to filter frequency
    :param downsampling: Downsampling in which to filter frequency
    :param keep: Whether to keep rows passing this frequency (passed to filter_rows)
    :param adj: Whether to use adj frequency
    :return: Filtered MatrixTable or Table
    """
    if frequency is None and allele_count is None:
        raise ValueError("At least one of frequency or allele_count must be specified")
    if direction not in ("above", "below", "equal"):
        raise ValueError('direction needs to be one of "above", "below", or "equal"')
    group = "adj" if adj else "raw"
    criteria = [lambda f: f.meta.get("group", "") == group]
    if frequency is not None:
        if direction == "above":
            criteria.append(lambda f: f.AF[1] > frequency)
        elif direction == "below":
            criteria.append(lambda f: f.AF[1] < frequency)
        else:
            criteria.append(lambda f: f.AF[1] == frequency)
    if allele_count is not None:
        if direction == "above":
            criteria.append(lambda f: f.AC[1] > allele_count)
        elif direction == "below":
            criteria.append(lambda f: f.AC[1] < allele_count)
        else:
            criteria.append(lambda f: f.AC[1] == allele_count)
    size = 1
    if population:
        criteria.append(lambda f: f.meta.get("pop", "") == population)
        size += 1
    if subpop:
        criteria.append(lambda f: f.meta.get("subpop", "") == subpop)
        size += 1
        # If one supplies a subpop but not a population, this will ensure this
        # gets it right
        if not population:
            size += 1
    if downsampling:
        criteria.append(lambda f: f.meta.get("downsampling", "") == str(downsampling))
        size += 1
        if not population:
            size += 1
            criteria.append(lambda f: f.meta.get("pop", "") == "global")
        if subpop:
            raise ValueError("No downsampling data for subpopulations implemented")
    criteria.append(lambda f: f.meta.size() == size)

    filt = lambda x: combine_functions(criteria, x)
    criteria = hl.any(filt, t.freq)
    return (
        t.filter_rows(criteria, keep=keep)
        if isinstance(t, hl.MatrixTable)
        else t.filter(criteria, keep=keep)
    )


def combine_functions(
    func_list: List[Callable[[bool], bool]],
    x: hl.expr.StructExpression,
    operator_func: Callable[[bool, bool], bool] = operator.iand,
) -> bool:
    """
    Combine a list of boolean functions to an Expression using the specified operator.

    .. note::

        The `operator_func` is applied cumulatively from left to right of the `func_list`.

    :param func_list: A list of boolean functions that can be applied to `x`.
    :param x: Expression to be passed to each function in `func_list`.
    :param operator_func: Operator function to combine the functions in `func_list`. Default is `operator.iand`.
    :return: A boolean from the combined operations.
    """
    cond = func_list[0](x)
    for c in func_list[1:]:
        cond = operator_func(cond, c(x))
    return cond


def low_conf_regions_expr(
    locus_expr: hl.expr.LocusExpression,
    filter_lcr: bool = True,
    filter_decoy: bool = True,
    filter_segdup: bool = True,
    filter_exome_low_coverage_regions: bool = False,
    filter_telomeres_and_centromeres: bool = False,
    high_conf_regions: Optional[List[str]] = None,
) -> hl.expr.BooleanExpression:
    """
    Create an expression to filter low confidence regions.

    :param locus_expr: Locus expression to use for filtering.
    :param filter_lcr: Whether to filter LCR regions
    :param filter_decoy: Whether to filter decoy regions
    :param filter_segdup: Whether to filter Segdup regions
    :param filter_exome_low_coverage_regions: Whether to filter exome low confidence regions
    :param filter_telomeres_and_centromeres: Whether to filter telomeres and centromeres
    :param high_conf_regions: Paths to set of high confidence regions to restrict to (union of regions)
    :return: Bool expression of whether loci are not low confidence (TRUE) or low confidence (FALSE)
    """
    build = get_reference_genome(locus_expr).name
    if build == "GRCh37":
        import gnomad.resources.grch37.reference_data as resources
    elif build == "GRCh38":
        import gnomad.resources.grch38.reference_data as resources
    else:
        raise ValueError(f"Unsupported reference genome build: {build}")

    criteria = []
    if filter_lcr:
        lcr = resources.lcr_intervals.ht()
        criteria.append(hl.is_missing(lcr[locus_expr]))

    if filter_decoy:
        decoy = resources.decoy_intervals.ht()
        criteria.append(hl.is_missing(decoy[locus_expr]))

    if filter_segdup:
        segdup = resources.seg_dup_intervals.ht()
        criteria.append(hl.is_missing(segdup[locus_expr]))

    if filter_exome_low_coverage_regions:
        high_cov = resources.high_coverage_intervals.ht()
        criteria.append(hl.is_missing(high_cov[locus_expr]))

    if filter_telomeres_and_centromeres:
        if build != "GRCh38":
            raise DataException(
                "The telomeres_and_centromeres resource only exists for GRCh38"
            )

        telomeres_and_centromeres = resources.telomeres_and_centromeres.ht()
        criteria.append(hl.is_missing(telomeres_and_centromeres[locus_expr]))

    if high_conf_regions is not None:
        for region in high_conf_regions:
            region = hl.import_locus_intervals(region)
            criteria.append(hl.is_defined(region[locus_expr]))

    if criteria:
        filter_criteria = functools.reduce(operator.iand, criteria)
        return filter_criteria
    else:
        raise ValueError("No low confidence regions requested for filtering.")


def filter_low_conf_regions(
    t: Union[hl.MatrixTable, hl.Table],
    **kwargs,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter low-confidence regions.

    :param t: MatrixTable or Table to filter.
    :param kwargs: Keyword arguments to pass to `low_conf_regions_expr`.
    :return: MatrixTable or Table with low confidence regions removed.
    """
    filter_criteria = low_conf_regions_expr(t.locus, **kwargs)
    if isinstance(t, hl.MatrixTable):
        t = t.filter_rows(filter_criteria)
    else:
        t = t.filter(filter_criteria)

    return t


def filter_to_autosomes(
    t: Union[hl.MatrixTable, hl.Table],
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter the Table or MatrixTable to autosomes only.

    This assumes that the input contains a field named `locus` of type Locus

    :param t: Input MT/HT
    :return:  MT/HT autosomes
    """
    reference = get_reference_genome(t.locus)
    autosomes = hl.parse_locus_interval(
        f"{reference.contigs[0]}-{reference.contigs[21]}", reference_genome=reference
    )
    return hl.filter_intervals(t, [autosomes])


def add_filters_expr(
    filters: Dict[str, hl.expr.BooleanExpression],
    current_filters: hl.expr.SetExpression = None,
) -> hl.expr.SetExpression:
    """
    Create an expression to create or add filters.

    For each entry in the `filters` dictionary, if the value evaluates to `True`,
    then the key is added as a filter name.

    Current filters are kept if provided using `current_filters`

    :param filters: The filters and their expressions
    :param current_filters: The set of current filters
    :return: An expression that can be used to annotate the filters
    """
    if current_filters is None:
        current_filters = hl.empty_set(hl.tstr)

    return hl.fold(
        lambda x, y: x.union(y),
        current_filters,
        [
            hl.if_else(filter_condition, hl.set([filter_name]), hl.empty_set(hl.tstr))
            for filter_name, filter_condition in filters.items()
        ],
    )


def subset_samples_and_variants(
    mtds: Union[hl.MatrixTable, hl.vds.VariantDataset],
    sample_path: str,
    header: bool = True,
    table_key: str = "s",
    sparse: bool = False,
    gt_expr: str = "GT",
    remove_dead_alleles: bool = False,
) -> Union[hl.MatrixTable, hl.vds.VariantDataset]:
    """
    Subset the MatrixTable or VariantDataset to the provided list of samples and their variants.

    :param mtds: Input MatrixTable or VariantDataset
    :param sample_path: Path to a file with list of samples
    :param header: Whether file with samples has a header. Default is True
    :param table_key: Key to sample Table. Default is "s"
    :param sparse: Whether the MatrixTable is sparse. Default is False
    :param gt_expr: Name of field in MatrixTable containing genotype expression. Default is "GT"
    :param remove_dead_alleles: Remove alleles observed in no samples. This option is currently only relevant when `mtds` is a VariantDataset. Default is False
    :return: MatrixTable or VariantDataset subsetted to specified samples and their variants
    """
    sample_ht = hl.import_table(sample_path, no_header=not header, key=table_key)
    sample_count = sample_ht.count()
    is_vds = isinstance(mtds, hl.vds.VariantDataset)
    if is_vds:
        mt = mtds.variant_data
    else:
        if remove_dead_alleles:
            raise ValueError(
                "Removal of alleles observed in no samples is currently only"
                " implemented when the input dataset is a VariantDataset."
            )
        mt = mtds
    missing_ht = sample_ht.anti_join(mt.cols())
    missing_ht_count = missing_ht.count()
    full_count = mt.count_cols()

    if missing_ht_count != 0:
        missing_samples = missing_ht.s.collect()
        raise DataException(
            f"Only {sample_count - missing_ht_count} out of"
            f" {sample_count} subsetting-table IDs matched IDs in the"
            f" {'VariantDataset' if is_vds else 'MatrixTable'}.\nIDs that aren't in the"
            f" MT: {missing_samples}\n"
        )

    if is_vds:
        mtds = hl.vds.filter_samples(
            mtds, sample_ht, keep=True, remove_dead_alleles=remove_dead_alleles
        )
        n_cols = mtds.variant_data.count_cols()
    else:
        mtds = mtds.semi_join_cols(sample_ht)
        if sparse:
            mtds = mtds.filter_rows(
                hl.agg.any(mtds[gt_expr].is_non_ref() | hl.is_defined(mtds.END))
            )
        else:
            mtds = mtds.filter_rows(hl.agg.any(mtds[gt_expr].is_non_ref()))
        n_cols = mtds.count_cols()

    logger.info(
        "Finished subsetting samples. Kept %d out of %d samples in %s",
        n_cols,
        full_count,
        "VariantDataset" if is_vds else "MatrixTable",
    )
    return mtds


def filter_to_clinvar_pathogenic(
    t: Union[hl.MatrixTable, hl.Table],
    clnrevstat_field: str = "CLNREVSTAT",
    clnsig_field: str = "CLNSIG",
    clnsigconf_field: str = "CLNSIGCONF",
    remove_no_assertion: bool = True,
    remove_conflicting: bool = True,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Return a MatrixTable or Table that filters the clinvar data to pathogenic and likely pathogenic variants.

    Example use:

    .. code-block:: python

        from gnomad.resources.grch38.reference_data import clinvar
        clinvar_ht = clinvar.ht()
        clinvar_ht = filter_to_clinvar_pathogenic(clinvar_ht)

    :param: t: Input dataset that contains clinvar data, could either be a MatrixTable or Table.
    :param clnrevstat_field: The field string for the expression that contains the review status of the clinical significance of clinvar variants.
    :param clnsig_field: The field string for the expression that contains the clinical signifcance of the clinvar variant.
    :param clnsigconf_field: The field string for the expression that contains the conflicting clinical significance values for the variant. For variants with no conflicting significance, this field should be undefined.
    :param remove_no_assertion: Flag for removing entries in which the clnrevstat (clinical significance) has no assertions (zero stars).
    :param remove_conflicting: Flag for removing entries with conflicting clinical interpretations.
    :return: Filtered MatrixTable or Table
    """
    logger.info(
        "Found %d variants before filtering",
        t.count_rows() if isinstance(t, hl.MatrixTable) else t.count(),
    )
    path_expr = (
        t.info[clnsig_field]
        .map(lambda x: x.lower())
        .map(lambda x: x.contains("pathogenic"))
        .any(lambda x: x)
    )

    if remove_no_assertion:
        logger.info("Variants without assertions will be removed.")
        no_star_assertions = hl.literal(
            {
                "no_assertion_provided",
                "no_assertion_criteria_provided",
                "no_interpretation_for_the_individual_variant",
            }
        )
        path_expr = path_expr & (
            hl.set(t.info[clnrevstat_field]).intersection(no_star_assertions).length()
            == 0
        )

    if remove_conflicting:
        logger.info(
            "Variants with conflicting clinical interpretations will be removed."
        )
        path_expr = path_expr & hl.is_missing(t.info[clnsigconf_field])

    if isinstance(t, hl.MatrixTable):
        t = t.filter_rows(path_expr)
    else:
        t = t.filter(path_expr)

    logger.info(
        "Found %d variants after filtering to clinvar pathogenic variants.",
        t.count_rows() if isinstance(t, hl.MatrixTable) else t.count(),
    )
    return t


def filter_gencode_ht(
    gencode_ht: Optional[hl.Table] = None,
    reference_genome: Optional[str] = "GRCh38",
    version: Optional[str] = None,
    protein_coding: bool = False,
    feature: Union[str, List[str]] = None,
    genes: Optional[Union[str, List[str]]] = None,
    by_gene_symbol: bool = True,
) -> hl.Table:
    """
    Filter a Gencode Table to specified criteria.

    .. note::

        If no Gencode Table is provided, a `reference_genome` Gencode Table resource
        will be used. If `version` is not provided, the default version of the Gencode
        Table resource will be used.

    :param gencode_ht: Gencode Table to use for filtering the input Table/MatrixTable
        to CDS regions. Default is None, which will use the default version of the
        Gencode Table resource.
    :param reference_genome: Reference genome build of Gencode Table to use if none is
        provided. Default is "GRCh38".
    :param version: Version of the Gencode Table to use if none is provided. Default is
        None.
    :param protein_coding: Whether to filter to only intervals where "transcript_type"
        is "protein_coding". Default is False.
    :param feature: Optional feature(s) to filter to. Can be a single feature string or
        list of features. Default is None.
    :param genes: Optional gene(s) to filter to. Can be a single gene string or list of
        genes. Default is None.
    :param by_gene_symbol: Whether to filter by gene symbol. Default is True. If False,
        will filter by gene ID.
    :return: Gencode Table filtered to specified criteria.
    """
    if gencode_ht is None and reference_genome is None:
        raise ValueError("Must provide a Gencode Table or reference genome build.")

    if gencode_ht is None:
        if reference_genome == "GRCh37":
            from gnomad.resources.grch37.reference_data import gencode
        elif reference_genome == "GRCh38":
            from gnomad.resources.grch38.reference_data import gencode
        else:
            raise ValueError(f"Unsupported reference genome build: {reference_genome}")

        if version is not None:
            gencode_ht = gencode.versions[version].ht()
        else:
            logger.info(
                "No Gencode Table or version was supplied, using Gencode version %s",
                gencode.default_version,
            )
            gencode_ht = gencode.ht()

    filter_expr = hl.literal(True)
    if feature:
        feature = [feature] if isinstance(feature, str) else feature
        filter_expr &= hl.literal(feature).contains(gencode_ht.feature)

    if protein_coding:
        filter_expr &= gencode_ht.transcript_type == "protein_coding"

    if genes:
        genes = hl.literal(
            [genes.upper()] if isinstance(genes, str) else [g.upper() for g in genes]
        )
        if by_gene_symbol:
            gene_field = "gene_name" if by_gene_symbol else "gene_id"
            filter_expr &= genes.contains(gencode_ht[gene_field])

    return gencode_ht.filter(filter_expr)


def filter_by_intervals(
    t: Union[hl.MatrixTable, hl.Table],
    intervals: Union[
        str,
        List[str],
        hl.expr.IntervalExpression,
        hl.Interval,
        List[hl.Interval],
    ],
    padding_bp: int = 0,
    max_collect_intervals: int = 3000,
    reference_genome: Optional[str] = None,
) -> hl.Table:
    """
    Filter Table/MatrixTable by interval(s).

    :param t: Input Table/MatrixTable to filter.
    :param intervals: Interval(s) to filter by. Can be a string, list of strings,
        IntervalExpression, Interval, or list of Intervals. If a string or list of
        strings, the interval string format has to be "contig:start-end",
        e.g.,"1:1000-2000" (GRCh37) or "chr1:1000-2000" (GRCh38).
    :param padding_bp: Number of bases to pad the intervals by. Default is 0.
    :param max_collect_intervals: Maximum number of intervals for the use of
        `hl.filter_intervals` for filtering. When the number of intervals to filter is
        greater than this number, `filter`/`filter_rows` will be used instead. The
        reason for this is that `hl.filter_intervals` is faster, but when the
        number of intervals is too large, this can cause memory errors. Default is
        3000.
    :param reference_genome: Reference genome build to use for parsing the intervals
        if the intervals are strings. Default is None.
    :return: Table/MatrixTable filtered by interval(s).
    """
    is_expr = isinstance(intervals, hl.expr.IntervalExpression)
    is_list = isinstance(intervals, list)

    if isinstance(intervals, str) or (is_list and isinstance(intervals[0], str)):
        intervals = parse_locus_intervals(intervals, reference_genome=reference_genome)

    intervals = pad_intervals(intervals, padding_bp) if padding_bp else intervals

    if not is_expr and not is_list:
        intervals = [intervals]

    if is_expr:
        _ht = intervals._indices.source
        num_intervals = _ht.count()

        # Only collect intervals if there are less than or equal to
        # `max_collect_intervals` to avoid memory issues.
        if num_intervals <= max_collect_intervals:
            logger.info(
                "Since %d is less than or equal to 'max_collect_intervals', "
                "collecting all intervals...",
                num_intervals,
            )
            intervals = intervals.collect()
        else:
            if padding_bp:
                _ht = _ht.key_by(padded_interval=intervals)

            return (
                t.filter_rows(hl.is_defined(_ht[t.locus]))
                if isinstance(t, hl.MatrixTable)
                else t.filter(hl.is_defined(_ht[t.locus]))
            )

    return hl.filter_intervals(t, intervals)


def filter_by_gencode_intervals(
    t: Union[hl.MatrixTable, hl.Table],
    gencode_ht: Optional[hl.Table] = None,
    protein_coding: bool = False,
    feature: Union[str, List[str]] = None,
    genes: Optional[Union[str, List[str]]] = None,
    by_gene_symbol: bool = True,
    padding_bp: int = 0,
    max_collect_intervals: int = 3000,
) -> hl.Table:
    """
    Filter a Table/MatrixTable based on Gencode Table annotations.

    .. note::

        If no Gencode Table is provided, the default version of the Gencode Table
        resource for the genome build of the input Table/MatrixTable will be used.

    :param t: Input Table/MatrixTable to filter.
    :param gencode_ht: Gencode Table to use for filtering the input Table/MatrixTable.
        Default is None, which will use the default version of the Gencode Table
        resource.
    :param protein_coding: Whether to filter to only intervals where "transcript_type"
        is "protein_coding". Default is False.
    :param feature: Optional feature(s) to filter to. Can be a single feature string or
        list of features. Default is None.
    :param genes: Optional gene(s) to filter to. Can be a single gene string or list of
        genes. Default is None.
    :param by_gene_symbol: Whether to filter by gene symbol. Default is True. If False,
        will filter by gene ID.
    :param padding_bp: Number of bases to pad the CDS intervals by. Default is 0.
    :param max_collect_intervals: Maximum number of intervals for the use of
         `hl.filter_intervals` for filtering. When the number of intervals to filter is
         greater than this number, `filter`/`filter_rows` will be used instead. The
         reason for this is that `hl.filter_intervals` is faster, but when the
         number of intervals is too large, this can cause memory errors. Default is
         3000.
    :return: Table/MatrixTable filtered to loci in requested Gencode intervals.
    """
    return filter_by_intervals(
        t,
        filter_gencode_ht(
            gencode_ht=gencode_ht,
            reference_genome=get_reference_genome(t.locus).name,
            protein_coding=protein_coding,
            feature=feature,
            genes=genes,
            by_gene_symbol=by_gene_symbol,
        ).interval,
        padding_bp=padding_bp,
        max_collect_intervals=max_collect_intervals,
    )


def filter_to_gencode_cds(
    t: Union[hl.MatrixTable, hl.Table],
    **kwargs,
) -> hl.Table:
    """
    Filter a Table/MatrixTable to only Gencode CDS regions in protein coding transcripts.

    Example use:

    .. code-block:: python

        from gnomad.resources.grch37.reference_data import gencode
        gencode_ht = gencode.ht()
        gencode_ht = filter_gencode_to_cds(gencode_ht)

    .. note::

        If no Gencode Table is provided, the default version of the Gencode Table
        resource for the genome build of the input Table/MatrixTable will be used.

    .. warning::

        This Gencode CDS interval filter does not take into account the
        transcript_id, it filters to any locus that is found in a CDS interval for
        any protein coding transcript. Therefore, if downstream analyses require
        filtering to CDS intervals by transcript, an additional step must be taken.
        For example, when filtering VEP transcript consequences, there may be cases
        where a variant is retained with this filter, but is considered outside the
        CDS intervals of the transcript per the VEP predicted consequence of the
        variant.

    :param t: Input Table/MatrixTable to filter.
    :param kwargs: Additional Keyword arguments to pass to `filter_gencode_ht`.
    :return: Table/MatrixTable filtered to loci in Gencode CDS intervals.
    """
    logger.warning(
        "This Gencode CDS interval filter does not filter by transcript! Please see the"
        " documentation for more details to confirm it's being used as intended."
    )
    return filter_by_gencode_intervals(t, protein_coding=True, feature="CDS", **kwargs)


def remove_fields_from_constant(
    constant: List[str], fields_to_remove: List[str]
) -> List[str]:
    """
    Remove fields from a list and display any field(s) missing from the original list.

    :param constant: List of fields
    :param fields_to_remove: List of fields to remove from `constant`
    """
    for field in fields_to_remove:
        if field in constant:
            constant.remove(field)
        else:
            logger.info("%s missing from %s", field, constant)

    return constant


def filter_x_nonpar(
    t: Union[hl.Table, hl.MatrixTable],
) -> Union[hl.Table, hl.MatrixTable]:
    """
    Filter to loci that are in non-PAR regions on chromosome X.

    :param t: Input Table or MatrixTable.
    :return: Filtered Table or MatrixTable.
    """
    rg = t.locus.dtype.reference_genome
    t = hl.filter_intervals(
        t,
        [
            hl.parse_locus_interval(contig, reference_genome=rg.name)
            for contig in rg.x_contigs
        ],
    )
    non_par_expr = t.locus.in_x_nonpar()

    return (
        t.filter(non_par_expr)
        if isinstance(t, hl.Table)
        else t.filter_rows(non_par_expr)
    )


def filter_y_nonpar(
    t: Union[hl.Table, hl.MatrixTable],
) -> Union[hl.Table, hl.MatrixTable]:
    """
    Filter to loci that are in non-PAR regions on chromosome Y.

    :param t: Input Table or MatrixTable.
    :return: Filtered Table or MatrixTable.
    """
    rg = t.locus.dtype.reference_genome
    t = hl.filter_intervals(
        t,
        [
            hl.parse_locus_interval(contig, reference_genome=rg.name)
            for contig in rg.y_contigs
        ],
    )
    non_par_expr = t.locus.in_y_nonpar()

    return (
        t.filter(non_par_expr)
        if isinstance(t, hl.Table)
        else t.filter_rows(non_par_expr)
    )


def filter_by_numeric_expr_range(
    t: Union[hl.MatrixTable, hl.Table],
    filter_expr: hl.NumericExpression,
    filter_range: tuple,
    keep_between: bool = True,
    inclusive: bool = True,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter rows in the Table/MatrixTable based on the range of a numeric expression.

    :param t: Input Table/MatrixTable.
    :param filter_expr: NumericExpression to apply `filter_range` to.
    :param filter_range: Range of values to apply to `filter_expr`.
    :param keep_between: Whether to keep the values between `filter_range` instead of keeping values outside `filter_range`. Default is True.
    :param inclusive: Whether or not to include the `filter_range` values themselves. Default is True.
    :return: Table/MatrixTable filtered to rows with specified criteria.
    """
    if inclusive and keep_between or not inclusive and not keep_between:
        criteria = (filter_expr >= filter_range[0]) & (filter_expr <= filter_range[1])
    else:
        criteria = (filter_expr > filter_range[0]) & (filter_expr < filter_range[1])

    if isinstance(t, hl.MatrixTable):
        return t.filter_rows(criteria, keep=keep_between)
    else:
        return t.filter(criteria, keep=keep_between)


def filter_for_mu(
    ht: hl.Table, gerp_lower_cutoff: float = -3.9885, gerp_upper_cutoff: float = 2.6607
) -> hl.Table:
    """
    Filter to non-coding annotations and remove GERP outliers.

    .. note::

        Values for `gerp_lower_cutoff` and `gerp_upper_cutoff` default to -3.9885 and
        2.6607, respectively. These values were precalculated on the GRCh37 context
        table and define the 5th and 95th percentiles.

    :param ht: Input Table.
    :param gerp_lower_cutoff: Minimum GERP score for variant to be included. Default is -3.9885.
    :param gerp_upper_cutoff: Maximum GERP score for variant to be included. Default is 2.6607.
    :return: Table filtered to intron or intergenic variants with GERP outliers removed.
    """
    ht = filter_by_numeric_expr_range(
        ht,
        filter_expr=ht.gerp,
        filter_range=(gerp_lower_cutoff, gerp_upper_cutoff),
        keep_between=True,
        inclusive=False,
    )
    ht = ht.filter(
        (ht.vep.most_severe_consequence == "intron_variant")
        | (ht.vep.most_severe_consequence == "intergenic_variant")
    )

    return ht


def split_vds_by_strata(
    vds: hl.vds.VariantDataset, strata_expr: hl.expr.Expression
) -> Dict[str, hl.vds.VariantDataset]:
    """
    Split a VDS into multiple VDSs based on `strata_expr`.

    :param vds: Input VDS.
    :param strata_expr: Expression on VDS variant_data MT to split on.
    :return: Dictionary where strata value is key and VDS is value.
    """
    vmt = vds.variant_data
    s_by_strata = vmt.aggregate_cols(
        hl.agg.group_by(strata_expr, hl.agg.collect_as_set(vmt.s))
    )

    return {
        strata: hl.vds.filter_samples(vds, list(s)) for strata, s in s_by_strata.items()
    }


def filter_meta_array(
    meta_expr: hl.expr.ArrayExpression,
    keys_to_keep: List[str] = None,
    keys_to_exclude: List[str] = None,
    key_value_pairs_to_keep: Dict[str, List[str]] = None,
    key_value_pairs_to_exclude: Dict[str, List[str]] = None,
    keep_combine_operator: str = "and",
    exclude_combine_operator: str = "and",
    combine_operator: str = "and",
    exact_match: bool = False,
) -> hl.expr.ArrayExpression:
    """
    Filter a metadata array expression based on keys and key-value pairs to keep/exclude.

    If `exact_match` is True, the filtering will only be applied to items with exactly
    the specified keys in `keys_to_keep` (and the keys in `key_value_pairs_to_keep`
    if provided). When `key_value_pairs_to_keep` is also provided, the keys in
    `key_value_pairs_to_keep` must also be present in the metadata item. This
    parameter is only relevant when `keys_to_keep` is provided, `combine_operator`
    is "and", and `exact_match` is True.

    :param meta_expr: Metadata array expression to filter.
    :param keys_to_keep: List of keys to keep.
    :param keys_to_exclude: List of keys to exclude.
    :param key_value_pairs_to_keep: Dictionary of key-value pairs to keep.
    :param key_value_pairs_to_exclude: Dictionary of key-value pairs to exclude.
    :param keep_combine_operator: Whether to use "and" or "or" to combine the filtering
        criteria for keys/key-value pairs to keep.
    :param exclude_combine_operator: Whether to use "and" or "or" to combine the
        filtering criteria for keys/key-value pairs to exclude.
    :param combine_operator: Whether to use "and" or "or" to combine the keep and
        exclude filtering criteria.
    :param exact_match: Whether to apply the filtering only to items with exactly the
        specified keys.
    :return: The filtered metadata array expression.
    """
    keys_to_keep = keys_to_keep or {}
    key_value_pairs_to_keep = key_value_pairs_to_keep or {}
    keys_to_exclude = keys_to_exclude or {}
    key_value_pairs_to_exclude = key_value_pairs_to_exclude or {}

    combine_operator_map = {"and": hl.all, "or": hl.any}
    for o in [keep_combine_operator, exclude_combine_operator, combine_operator]:
        if o not in combine_operator_map:
            raise ValueError(
                "The combine operators must be one of 'and' or 'or', but found" f" {o}!"
            )

    # Assign operators to their respective values in the combine_operator_map dict.
    keep_combine_operator = combine_operator_map[keep_combine_operator]
    exclude_combine_operator = combine_operator_map[exclude_combine_operator]
    combine_operator = combine_operator_map[combine_operator]

    def _get_filter(m: hl.DictExpression) -> hl.expr.BooleanExpression:
        """
        Get the filter to apply to the metadata item.

        :param m: Metadata item.
        :return: Filter to apply to the metadata item.
        """
        # If keys_to_keep is provided, filter to only metadata items with the specified
        # keys. If exact_match is True, filter to only metadata items with the exact
        # keys specified in keys_to_keep, where any keys in key_value_pairs_to_keep
        # are also present.
        if exact_match:
            keep_filter = [
                hl.set(set(keys_to_keep) | set(key_value_pairs_to_keep.keys()))
                == hl.set(m.keys())
            ]
        else:
            keep_filter = [m.contains(k) for k in keys_to_keep]

        # If key_value_pairs_to_keep is provided, filter to only metadata items with the
        # specified key-value pairs.
        keep_filter += [
            hl.literal(v if isinstance(v, list) else [v]).contains(m.get(k, ""))
            for k, v in key_value_pairs_to_keep.items()
        ]

        # If keys_to_exclude is provided, filter to only metadata items without the
        # specified keys and if key_value_pairs_to_exclude is provided, filter to only
        # metadata items without the specified key-value pairs.
        exclude_filter = [~m.contains(k) for k in keys_to_exclude] + [
            ~hl.literal(v if isinstance(v, list) else [v]).contains(m.get(k, ""))
            for k, v in key_value_pairs_to_exclude.items()
        ]

        filters = []
        if keep_filter:
            filters.append(keep_combine_operator(keep_filter))
        if exclude_filter:
            filters.append(exclude_combine_operator(exclude_filter))

        return combine_operator(filters)

    return meta_expr.filter(lambda m: _get_filter(m))


def filter_arrays_by_meta(
    meta_expr: hl.expr.ArrayExpression,
    meta_indexed_exprs: Union[
        Dict[str, hl.expr.ArrayExpression], hl.expr.ArrayExpression
    ],
    items_to_filter: Union[
        List[str], Dict[str, Union[List[str], Dict[str, Union[List[str], bool]]]]
    ],
    keep: bool = True,
    keep_combine_operator: str = "and",
    exclude_combine_operator: str = "and",
    combine_operator: str = "and",
    exact_match: bool = False,
) -> Tuple[
    hl.expr.ArrayExpression,
    Union[Dict[str, hl.expr.ArrayExpression], hl.expr.ArrayExpression],
]:
    """
    Filter both metadata array expression and metadata indexed expression by `items_to_filter`.

    The `items_to_filter` can be used to filter in the following ways based on
    `meta_expr` items:

        - By a list of keys, e.g. ``["sex", "downsampling"]``.
        - By specific key: value pairs, e.g. to filter where 'pop' is 'han' or 'papuan'
          ``{"pop": ["han", "papuan"]}``, or where 'pop' is 'afr' and/or 'sex' is 'XX'
          ``{"pop": ["afr"], "sex": ["XX"]}``.
        - By specific key: value pairs with differing keep values, e.g.:

            .. code-block:: python

                {
                    "gen_anc": {"values": ["global", "afr"], "keep": True},
                    "downsampling": {"keep": True},
                    "subset": {"keep": False},
                }

    The items can be kept or removed from `meta_indexed_expr` and `meta_expr` based on
    the value of `keep`. For example if `meta_indexed_exprs` is {'freq': ht.freq,
    'freq_meta_sample_count': ht.index_globals().freq_meta_sample_count} and `meta_expr`
    is ht.freq_meta then if `keep` is True, the items specified by `items_to_filter`
    such as  'pop' = 'han' will be kept and all other items will be removed from the
    ht.freq, ht.freq_meta_sample_count, and ht.freq_meta. `meta_indexed_exprs` can also
    be a single array expression such as ht.freq.

    The filtering can also be applied such that all criteria must be met
    (`combine_operator` = "and") by the `meta_expr` item in order to be filtered,
    or at least one of the specified criteria must be met (`combine_operator` = "or")
    by the `meta_expr` item in order to be filtered.

    The `exact_match` parameter can be used to apply the `keep` parameter to only items
    specified in the `items_to_filter` parameter. For example, by default, if `keep` is
    True, `combine_operator` is "and", and `items_to_filter` is ["sex", "downsampling"],
    then all items in `meta_expr` with both "sex" and "downsampling" as keys will be
    kept. However, if `exact_match` is True, then the items in `meta_expr` will only be
    kept if "sex" and "downsampling" are the only keys in the meta dict.

    :param meta_expr: Metadata expression that contains the values of the elements in
        `meta_indexed_expr`. The most often used expression is `freq_meta` to index into
        a 'freq' array.
    :param meta_indexed_exprs: Either a Dictionary where the keys are the expression name
        and the values are the expressions indexed by the `meta_expr` such as a 'freq'
        array or just a single expression indexed by the `meta_expr`.
    :param items_to_filter: Items to filter by, either a list or a dictionary.
    :param keep: Whether to keep or remove the items specified by `items_to_filter`.
    :param keep_combine_operator: Whether to use "and" or "or" to combine the filtering
        criteria for keys/key-value pairs to keep.
    :param exclude_combine_operator: Whether to use "and" or "or" to combine the
        filtering criteria for keys/key-value pairs to exclude.
    :param combine_operator: Whether to use "and" or "or" to combine the keep and
        exclude filtering criteria.
    :param exact_match: Whether to apply the `keep` parameter to only the items
        specified in the `items_to_filter` parameter or to all items in `meta_expr`.
        See the example above for more details. Default is False.
    :return: A Tuple of the filtered metadata expression and a dictionary of metadata
        indexed expressions when meta_indexed_expr is a Dictionary or a single filtered
        array expression when meta_indexed_expr is a single array expression.
    """
    meta_expr = meta_expr.collect(_localize=False)[0]

    # If only a single array expression needs to be filtered, make meta_indexed_exprs
    # a dictionary with a single key "_tmp" so it can be filtered in the same way as
    # a dictionary of array expressions.
    if isinstance(meta_indexed_exprs, hl.expr.ArrayExpression):
        meta_indexed_exprs = {"_tmp": meta_indexed_exprs}

    # If items_to_filter is a list, convert it to a dictionary with the key being the
    # item to filter and the value being None, so it can be filtered in the same way as
    # a dictionary of items to filter.
    if isinstance(items_to_filter, list):
        items_to_filter = {k: None for k in items_to_filter}
    elif isinstance(items_to_filter, dict):
        # If items_to_filter is a dictionary with lists as values, convert the lists
        # to dictionaries with the key "values" and the value being the list of values
        # to filter by.
        items_to_filter = {
            k: (
                v
                if v is None or isinstance(v, dict)
                else {"values": v if isinstance(v, list) else [v]}
            )
            for k, v in items_to_filter.items()
        }
    else:
        raise TypeError("items_to_filter must be a list or a dictionary!")

    # Use filter_meta_array to filter the meta_expr to keep only the items specified
    # by items_to_filter.
    keys_to_keep = []
    keys_to_exclude = []
    key_value_pairs_to_keep = {}
    key_value_pairs_to_exclude = {}

    for k, v in items_to_filter.items():
        # Set item_keep to 'keep' parameter if value is None or if 'keep' value is not
        # defined in that items' dictionary. Otherwise, (if already defined in the
        # item's dictionary), use the 'keep' value defined in the dictionary.
        item_keep = keep if v is None or "keep" not in v else v["keep"]

        if item_keep:
            if v is not None and "values" in v:
                key_value_pairs_to_keep[k] = v["values"]
            else:
                keys_to_keep.append(k)
        else:
            if v is not None and "values" in v:
                key_value_pairs_to_exclude[k] = v["values"]
            else:
                keys_to_exclude.append(k)

    filtered_meta_expr = filter_meta_array(
        meta_expr,
        keys_to_keep=keys_to_keep,
        keys_to_exclude=keys_to_exclude,
        key_value_pairs_to_keep=key_value_pairs_to_keep,
        key_value_pairs_to_exclude=key_value_pairs_to_exclude,
        keep_combine_operator=keep_combine_operator,
        exclude_combine_operator=exclude_combine_operator,
        combine_operator=combine_operator,
        exact_match=exact_match,
    )

    # Filter the enumerated meta_exprs to only keep the items that match the metadata
    # dictionaries in the filtered meta expression.
    filtered_meta_idx_expr = hl.enumerate(meta_expr).filter(
        lambda x: filtered_meta_expr.contains(x[1])
    )

    # Filter each of the array expressions in meta_indexed_exprs to only keep the items
    # that match the metadata dictionaries in the filtered meta expression.
    meta_indexed_exprs = {
        k: filtered_meta_idx_expr.map(lambda x: v[x[0]])
        for k, v in meta_indexed_exprs.items()
    }

    # If the original meta_indexed_exprs was a single array expression, return the
    # filtered meta_indexed_exprs as a single array expression.
    if "_tmp" in meta_indexed_exprs:
        meta_indexed_exprs = meta_indexed_exprs["_tmp"]

    return filtered_meta_expr, meta_indexed_exprs

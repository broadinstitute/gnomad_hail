# noqa: D100

import logging
from pprint import pprint
from typing import Any, Dict, List, Optional, Union

import hail as hl
from hail.utils.misc import new_temp_file

from gnomad.resources.grch38.gnomad import CURRENT_MAJOR_RELEASE, POPS, SEXES
from gnomad.utils.vcf import HISTS, SORT_ORDER, make_label_combos

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generic_field_check(
    ht: hl.Table,
    check_description: str,
    display_fields: hl.expr.StructExpression,
    cond_expr: hl.expr.BooleanExpression = None,
    verbose: bool = False,
    show_percent_sites: bool = False,
    n_fail: Optional[int] = None,
    ht_count: Optional[int] = None,
) -> None:
    """
    Check generic logical condition `cond_expr` involving annotations in a Hail Table when `n_fail` is absent and print the results to stdout.

    Displays the number of rows (and percent of rows, if `show_percent_sites` is True) in the Table that fail, either previously computed as `n_fail` or that match the `cond_expr`, and fail to be the desired condition (`check_description`).
    If the number of rows that match the `cond_expr` or `n_fail` is 0, then the Table passes that check; otherwise, it fails.

    .. note::

        `cond_expr` and `check_description` are opposites and should never be the same.
        E.g., If `cond_expr` filters for instances where the raw AC is less than adj AC,
        then it is checking sites that fail to be the desired condition (`check_description`)
        of having a raw AC greater than or equal to the adj AC.

    :param ht: Table containing annotations to be checked.
    :param check_description: String describing the condition being checked; is displayed in stdout summary message.
    :param display_fields: StructExpression containing annotations to be displayed in case of failure (for troubleshooting purposes); these fields are also displayed if verbose is True.
    :param cond_expr: Optional logical expression referring to annotations in ht to be checked.
    :param verbose: If True, show top values of annotations being checked, including checks that pass; if False, show only top values of annotations that fail checks.
    :param show_percent_sites: Show percentage of sites that fail checks. Default is False.
    :param n_fail: Optional number of sites that fail the conditional checks (previously computed). If not supplied, `cond_expr` is used to filter the Table and obtain the count of sites that fail the checks.
    :param ht_count: Optional number of sites within hail Table (previously computed). If not supplied, a count of sites in the Table is performed.
    :return: None
    """
    if n_fail is None and cond_expr is None:
        raise ValueError("At least one of n_fail or cond_expr must be defined!")

    if n_fail is None and cond_expr is not None:
        n_fail = ht.filter(cond_expr).count()

    if show_percent_sites and (ht_count is None):
        ht_count = ht.count()

    if n_fail > 0:
        logger.info("Found %d sites that fail %s check:", n_fail, check_description)
        if show_percent_sites:
            logger.info(
                "Percentage of sites that fail: %.2f %%", 100 * (n_fail / ht_count)
            )
        if cond_expr is not None:
            ht = ht.select(_fail=cond_expr, **display_fields)
            ht.filter(ht._fail).drop("_fail").show()
    else:
        logger.info("PASSED %s check", check_description)
        if verbose:
            ht.select(**display_fields).show()


def make_filters_expr_dict(
    ht: hl.Table,
    extra_filter_checks: Optional[Dict[str, hl.expr.Expression]] = None,
    variant_filter_field: str = "RF",
) -> Dict[str, hl.expr.Expression]:
    """
    Make Hail expressions to measure % variants filtered under varying conditions of interest.

    Checks for:
        - Total number of variants
        - Fraction of variants removed due to:
            - Any filter
            - Inbreeding coefficient filter in combination with any other filter
            - AC0 filter in combination with any other filter
            - `variant_filter_field` filtering in combination with any other filter
            - Only inbreeding coefficient filter
            - Only AC0 filter
            - Only filtering defined by `variant_filter_field`

    :param ht: Table containing 'filter' annotation to be examined.
    :param extra_filter_checks: Optional dictionary containing filter condition name (key) extra filter expressions (value) to be examined.
    :param variant_filter_field: String of variant filtration used in the filters annotation on `ht` (e.g. RF, VQSR, AS_VQSR). Default is "RF".
    :return: Dictionary containing Hail aggregation expressions to examine filter flags.
    """
    filters_dict = {
        "n": hl.agg.count(),
        "frac_any_filter": hl.agg.fraction(hl.len(ht.filters) != 0),
        "frac_inbreed_coeff": hl.agg.fraction(ht.filters.contains("InbreedingCoeff")),
        "frac_ac0": hl.agg.fraction(ht.filters.contains("AC0")),
        f"frac_{variant_filter_field.lower()}": hl.agg.fraction(
            ht.filters.contains(variant_filter_field)
        ),
        "frac_inbreed_coeff_only": hl.agg.fraction(
            ht.filters.contains("InbreedingCoeff") & (ht.filters.length() == 1)
        ),
        "frac_ac0_only": hl.agg.fraction(
            ht.filters.contains("AC0") & (ht.filters.length() == 1)
        ),
        f"frac_{variant_filter_field.lower()}_only": hl.agg.fraction(
            ht.filters.contains(variant_filter_field) & (ht.filters.length() == 1)
        ),
    }
    if extra_filter_checks:
        filters_dict.update(extra_filter_checks)

    return filters_dict


def make_group_sum_expr_dict(
    t: Union[hl.MatrixTable, hl.Table],
    subset: str,
    label_groups: Dict[str, List[str]],
    sort_order: List[str] = SORT_ORDER,
    delimiter: str = "-",
    metric_first_field: bool = True,
    metrics: List[str] = ["AC", "AN", "nhomalt"],
) -> Dict[str, Dict[str, Union[hl.expr.Int64Expression, hl.expr.StructExpression]]]:
    """
    Compute the sum of call stats annotations for a specified group of annotations, compare to the annotated version, and display the result in stdout.

    For example, if subset1 consists of pop1, pop2, and pop3, check that t.info.AC-subset1 == sum(t.info.AC-subset1-pop1, t.info.AC-subset1-pop2, t.info.AC-subset1-pop3).

    :param t: Input MatrixTable or Table containing call stats annotations to be summed.
    :param subset: String indicating sample subset.
    :param label_groups: Dictionary containing an entry for each label group, where key is the name of the grouping, e.g. "sex" or "pop", and value is a list of all possible values for that grouping (e.g. ["XY", "XX"] or ["afr", "nfe", "amr"]).
    :param sort_order: List containing order to sort label group combinations. Default is SORT_ORDER.
    :param delimiter: String to use as delimiter when making group label combinations. Default is "-".
    :param metric_first_field: If True, metric precedes subset in the Table's fields, e.g. AC-hgdp. If False, subset precedes metric, hgdp-AC. Default is True.
    :param metrics: List of metrics to sum and compare to annotationed versions. Default is ["AC", "AN", "nhomalt"].
    :return: Dictionary of sample sum field check expressions and display fields.
    """
    t = t.rows() if isinstance(t, hl.MatrixTable) else t

    # Check if subset string is provided to avoid adding a delimiter to empty string
    # (An empty string is passed to run this check on the entire callset)
    if subset:
        subset += delimiter

    label_combos = make_label_combos(label_groups, label_delimiter=delimiter)
    # Grab the first group for check and remove if from the label_group
    # dictionary. In gnomAD, this is 'adj', as we do not retain the raw metric
    # counts for all sample groups so we do not check raw sample sums.
    group = label_groups.pop("group")[0]
    # sum_group is a the type of high level annotation that you want to sum
    # e.g. 'pop', 'pop-sex', 'sex'.
    sum_group = delimiter.join(
        sorted(label_groups.keys(), key=lambda x: sort_order.index(x))
    )
    info_fields = t.info.keys()

    # Loop through metrics and the label combos to build a dictionary
    # where the key is a string representing the sum_group annotations and the value is the sum of these annotations.
    # If metric_first_field is True, metric is AC, subset is tgp, group is adj, sum_group is pop, then the values below are:
    # sum_group_exprs = ["AC-tgp-pop1", "AC-tgp-pop2", "AC-tgp-pop3"]
    # annot_dict = {'sum-AC-tgp-adj-pop': hl.sum(["AC-tgp-adj-pop1",
    # "AC-tgp-adj-pop2", "AC-tgp-adj-pop3"])}
    annot_dict = {}
    for metric in metrics:
        if metric_first_field:
            field_prefix = f"{metric}{delimiter}{subset}"
        else:
            field_prefix = f"{subset}{metric}{delimiter}"

        sum_group_exprs = []
        for label in label_combos:
            field = f"{field_prefix}{label}"
            if field in info_fields:
                sum_group_exprs.append(t.info[field])
            else:
                logger.warning("%s is not in table's info field", field)

        annot_dict[f"sum{delimiter}{field_prefix}{group}{delimiter}{sum_group}"] = (
            hl.sum(sum_group_exprs)
        )

    # If metric_first_field is True, metric is AC, subset is tgp, sum_group is pop, and group is adj, then the values below are:
    # check_field_left = "AC-tgp-adj"
    # check_field_right = "sum-AC-tgp-adj-pop" to match the annotation dict
    # key from above
    field_check_expr = {}
    for metric in metrics:
        if metric_first_field:
            check_field_left = f"{metric}{delimiter}{subset}{group}"
        else:
            check_field_left = f"{subset}{metric}{delimiter}{group}"
        check_field_right = f"sum{delimiter}{check_field_left}{delimiter}{sum_group}"
        field_check_expr[f"{check_field_left} = {check_field_right}"] = {
            "expr": t.info[check_field_left] != annot_dict[check_field_right],
            "agg_func": hl.agg.count_where,
            "display_fields": hl.struct(
                **{
                    check_field_left: t.info[check_field_left],
                    check_field_right: annot_dict[check_field_right],
                }
            ),
        }
    return field_check_expr


def compare_row_counts(ht1: hl.Table, ht2: hl.Table) -> bool:
    """
    Check if the row counts in two Tables are the same.

    :param ht1: First Table to be checked.
    :param ht2: Second Table to be checked.
    :return: Whether the row counts are the same.
    """
    r_count1 = ht1.count()
    r_count2 = ht2.count()
    logger.info("%d rows in left table; %d rows in right table", r_count1, r_count2)
    return r_count1 == r_count2


def summarize_variant_filters(
    t: Union[hl.MatrixTable, hl.Table],
    variant_filter_field: str = "RF",
    problematic_regions: List[str] = ["lcr", "segdup", "nonpar"],
    single_filter_count: bool = False,
    site_gt_check_expr: Dict[str, hl.expr.BooleanExpression] = None,
    extra_filter_checks: Optional[Dict[str, hl.expr.Expression]] = None,
    n_rows: int = 50,
    n_cols: int = 140,
) -> None:
    """
    Summarize variants filtered under various conditions in input MatrixTable or Table.

    Summarize counts for:
        - Total number of variants
        - Fraction of variants removed due to:
            - Any filter
            - Inbreeding coefficient filter in combination with any other filter
            - AC0 filter in combination with any other filter
            - `variant_filter_field` filtering in combination with any other filter in combination with any other filter
            - Only inbreeding coefficient filter
            - Only AC0 filter
            - Only `variant_filter_field` filtering

    :param t: Input MatrixTable or Table to be checked.
    :param variant_filter_field: String of variant filtration used in the filters annotation on `ht` (e.g. RF, VQSR, AS_VQSR). Default is "RF".
    :param problematic_regions: List of regions considered problematic to run filter check in. Default is ["lcr", "segdup", "nonpar"].
    :param single_filter_count: If True, explode the Table's filter column and give a supplement total count of each filter. Default is False.
    :param site_gt_check_expr: Optional dictionary of strings and boolean expressions typically used to log how many monoallelic or 100% heterozygous sites are in the Table.
    :param extra_filter_checks: Optional dictionary containing filter condition name (key) and extra filter expressions (value) to be examined.
    :param n_rows: Number of rows to display only when showing percentages of filtered variants grouped by multiple conditions. Default is 50.
    :param n_cols: Number of columns to display only when showing percentages of filtered variants grouped by multiple conditions. Default is 140.
    :return: None
    """
    t = t.rows() if isinstance(t, hl.MatrixTable) else t

    filters = t.aggregate(hl.agg.counter(t.filters))
    logger.info("Variant filter counts: %s", filters)

    if single_filter_count:
        exp_t = t.explode(t.filters)
        filters = exp_t.aggregate(hl.agg.counter(exp_t.filters))
        logger.info("Exploded variant filter counts: %s", filters)

    if site_gt_check_expr is not None:
        for k, m_expr in site_gt_check_expr.items():
            if isinstance(t, hl.MatrixTable):
                gt_check_sites = t.filter_rows(m_expr).count_rows()
            else:
                gt_check_sites = t.filter(m_expr).count()
            logger.info("There are %d %s sites in the dataset.", gt_check_sites, k)

    filtered_expr = hl.len(t.filters) > 0
    problematic_region_expr = hl.any(
        lambda x: x, [t.info[region] for region in problematic_regions]
    )

    t = t.annotate(
        is_filtered=filtered_expr, in_problematic_region=problematic_region_expr
    )

    def _filter_agg_order(
        t: Union[hl.MatrixTable, hl.Table],
        group_exprs: Dict[str, hl.expr.Expression],
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> None:
        """
        Perform validity checks to measure percentages of variants filtered under different conditions.

        :param t: Input MatrixTable or Table.
        :param group_exprs: Dictionary of expressions to group the Table by.
        :param extra_filter_checks: Optional dictionary containing filter condition name (key) and extra filter expressions (value) to be examined.
        :param n_rows: Number of rows to show. Default is None (to display 10 rows).
        :param n_cols: Number of columns to show. Default is None (to display 10 cols).
        :return: None
        """
        t = t.rows() if isinstance(t, hl.MatrixTable) else t
        # NOTE: make_filters_expr_dict returns a dict with %ages of variants filtered
        t.group_by(**group_exprs).aggregate(
            **make_filters_expr_dict(t, extra_filter_checks, variant_filter_field)
        ).order_by(hl.desc("n")).show(n_rows, n_cols)

    logger.info(
        "Checking distributions of filtered variants amongst variant filters..."
    )
    _filter_agg_order(t, {"is_filtered": t.is_filtered}, n_rows, n_cols)

    add_agg_expr = {}
    if "allele_type" in t.info:
        logger.info("Checking distributions of variant type amongst variant filters...")
        add_agg_expr["allele_type"] = t.info.allele_type
        _filter_agg_order(t, add_agg_expr, n_rows, n_cols)

    if "in_problematic_region" in t.row:
        logger.info(
            "Checking distributions of variant type and region type amongst variant"
            " filters..."
        )
        add_agg_expr["in_problematic_region"] = t.in_problematic_region
        _filter_agg_order(t, add_agg_expr, n_rows, n_cols)

    if "n_alt_alleles" in t.info:
        logger.info(
            "Checking distributions of variant type, region type, and number of alt alleles"
            " amongst variant filters..."
        )
        add_agg_expr["n_alt_alleles"] = t.info.n_alt_alleles
        _filter_agg_order(t, add_agg_expr, n_rows, n_cols)


def generic_field_check_loop(
    ht: hl.Table,
    field_check_expr: Dict[str, Dict[str, Any]],
    verbose: bool,
    show_percent_sites: bool = False,
    ht_count: int = None,
) -> None:
    """
    Loop through all conditional checks for a given hail Table.

    This loop allows aggregation across the hail Table once, as opposed to aggregating during every conditional check.

    :param ht: Table containing annotations to be checked.
    :param field_check_expr: Dictionary whose keys are conditions being checked and values are the expressions for filtering to condition.
    :param verbose: If True, show top values of annotations being checked, including checks that pass; if False, show only top values of annotations that fail checks.
    :param show_percent_sites: Show percentage of sites that fail checks. Default is False.
    :param ht_count: Previously computed sum of sites within hail Table. Default is None.
    :return: None
    """
    ht_field_check_counts = ht.aggregate(
        hl.struct(**{k: v["agg_func"](v["expr"]) for k, v in field_check_expr.items()})
    )
    for check_description, n_fail in ht_field_check_counts.items():
        generic_field_check(
            ht,
            check_description=check_description,
            n_fail=n_fail,
            display_fields=field_check_expr[check_description]["display_fields"],
            cond_expr=field_check_expr[check_description]["expr"],
            verbose=verbose,
            show_percent_sites=show_percent_sites,
            ht_count=ht_count,
        )


def compare_subset_freqs(
    t: Union[hl.MatrixTable, hl.Table],
    subsets: List[str],
    verbose: bool,
    show_percent_sites: bool = True,
    delimiter: str = "-",
    metric_first_field: bool = True,
    metrics: List[str] = ["AC", "AN", "nhomalt"],
) -> None:
    """
    Perform validity checks on frequency data in input Table.

    Check:
        - Number of sites where callset frequency is equal to a subset frequency (raw and adj)
            - eg. t.info.AC-adj != t.info.AC-subset1-adj
        - Total number of sites where the raw allele count annotation is defined

    :param t: Input MatrixTable or Table.
    :param subsets: List of sample subsets.
    :param verbose: If True, show top values of annotations being checked, including checks that pass; if False, show only top values of annotations that fail checks.
    :param show_percent_sites: If True, show the percentage and count of overall sites that fail; if False, only show the number of sites that fail.
    :param delimiter: String to use as delimiter when making group label combinations. Default is "-".
    :param metric_first_field: If True, metric precedes subset, e.g. AC-non_v2-. If False, subset precedes metric, non_v2-AC-XY. Default is True.
    :param metrics: List of metrics to compare between subset and entire callset. Default is ["AC", "AN", "nhomalt"].
    :return: None
    """
    t = t.rows() if isinstance(t, hl.MatrixTable) else t

    field_check_expr = {}
    for subset in subsets:
        if subset:
            for metric in metrics:
                for group in ["adj", "raw"]:
                    logger.info(
                        "Comparing the %s subset's %s %s to entire callset's %s %s",
                        subset,
                        group,
                        metric,
                        group,
                        metric,
                    )
                    check_field_left = f"{metric}{delimiter}{group}"
                    if metric_first_field:
                        check_field_right = (
                            f"{metric}{delimiter}{subset}{delimiter}{group}"
                        )
                    else:
                        check_field_right = (
                            f"{subset}{delimiter}{metric}{delimiter}{group}"
                        )

                    field_check_expr[f"{check_field_left} != {check_field_right}"] = {
                        "expr": t.info[check_field_left] == t.info[check_field_right],
                        "agg_func": hl.agg.count_where,
                        "display_fields": hl.struct(
                            **{
                                check_field_left: t.info[check_field_left],
                                check_field_right: t.info[check_field_right],
                            }
                        ),
                    }

    generic_field_check_loop(
        t,
        field_check_expr,
        verbose,
        show_percent_sites=show_percent_sites,
    )

    # Spot check the raw AC counts
    total_defined_raw_ac = t.aggregate(
        hl.agg.count_where(hl.is_defined(t.info[f"AC{delimiter}raw"]))
    )
    logger.info("Total defined raw AC count: %s", total_defined_raw_ac)


def sum_group_callstats(
    t: Union[hl.MatrixTable, hl.Table],
    sexes: List[str] = SEXES,
    subsets: List[str] = [""],
    pops: List[str] = POPS[CURRENT_MAJOR_RELEASE]["exomes"],
    groups: List[str] = ["adj"],
    additional_subsets_and_pops: Dict[str, List[str]] = None,
    verbose: bool = False,
    sort_order: List[str] = SORT_ORDER,
    delimiter: str = "-",
    metric_first_field: bool = True,
    metrics: List[str] = ["AC", "AN", "nhomalt"],
) -> None:
    """
    Compute the sum of annotations for a specified group of annotations, and compare to the annotated version.

    Displays results from checking the sum of the specified annotations in stdout.
    Also checks that annotations for all expected sample populations are present.

    :param t: Input Table.
    :param sexes: List of sexes in table.
    :param subsets: List of sample subsets that contain pops passed in pops parameter. An empty string, e.g. "", should be passed to test entire callset. Default is [""].
    :param pops: List of pops contained within the subsets. Default is POPS[CURRENT_MAJOR_RELEASE]["exomes"].
    :param groups: List of callstat groups, e.g. "adj" and "raw" contained within the callset. gnomAD does not store the raw callstats for the pop or sex groupings of any subset. Default is ["adj"]
    :param sample_sum_sets_and_pops: Dict with subset (keys) and list of the subset's specific populations (values). Default is None.
    :param verbose: If True, show top values of annotations being checked, including checks that pass; if False, show only top values of annotations that fail checks. Default is False.
    :param sort_order: List containing order to sort label group combinations. Default is SORT_ORDER.
    :param delimiter: String to use as delimiter when making group label combinations. Default is "-".
    :param metric_first_field: If True, metric precedes label group, e.g. AC-afr-male. If False, label group precedes metric, afr-male-AC. Default is True.
    :param metrics: List of metrics to sum and compare to annotationed versions. Default is ["AC", "AN", "nhomalt"].
    :return: None
    """
    # TODO: Add support for subpop sums
    t = t.rows() if isinstance(t, hl.MatrixTable) else t

    field_check_expr = {}
    default_pop_subset = {subset: pops for subset in subsets}
    sample_sum_sets_and_pops = (
        {**default_pop_subset, **additional_subsets_and_pops}
        if additional_subsets_and_pops
        else default_pop_subset
    )
    for subset, pops in sample_sum_sets_and_pops.items():
        for group in groups:
            field_check_expr_s = make_group_sum_expr_dict(
                t,
                subset,
                dict(group=[group], pop=pops),
                sort_order,
                delimiter,
                metric_first_field,
                metrics,
            )
            field_check_expr.update(field_check_expr_s)
            field_check_expr_s = make_group_sum_expr_dict(
                t,
                subset,
                dict(group=[group], sex=sexes),
                sort_order,
                delimiter,
                metric_first_field,
                metrics,
            )
            field_check_expr.update(field_check_expr_s)
            field_check_expr_s = make_group_sum_expr_dict(
                t,
                subset,
                dict(group=[group], pop=pops, sex=sexes),
                sort_order,
                delimiter,
                metric_first_field,
                metrics,
            )
            field_check_expr.update(field_check_expr_s)

    generic_field_check_loop(t, field_check_expr, verbose)


def summarize_variants(
    t: Union[hl.MatrixTable, hl.Table],
    expected_contigs: List[str] = None,
) -> hl.Struct:
    """
    Get summary of variants in a MatrixTable or Table.

    Print the number of variants to stdout and check that each chromosome has variant calls. If requested,
    check that all expected contigs are found in the variant summary and that no unexpected contigs are found.

    :param t: Input MatrixTable or Table to be checked.
    :param expected_contigs: List of contigs expected to be found in the input.
    :return: Struct of variant summary
    """
    if isinstance(t, hl.MatrixTable):
        logger.info("Dataset has %d samples.", t.count_cols())

    var_summary = hl.summarize_variants(t, show=False)
    logger.info(
        "Dataset has %d variants distributed across the following contigs: %s",
        var_summary.n_variants,
        var_summary.contigs,
    )

    # Check that all contigs have variant calls.
    for contig in var_summary.contigs:
        if var_summary.contigs[contig] == 0:
            logger.warning("%s has no variants called", var_summary.contigs)

    # Check that all expected contigs are found in the variant summary
    # and that no unexpected contigs are found.
    if expected_contigs:
        var_summary_contigs = var_summary["contigs"].keys()
        missing_contigs = expected_contigs - var_summary_contigs
        unexpected_contigs = var_summary_contigs - expected_contigs

        logger.info("Expected contigs: %s", expected_contigs)
        logger.info("Found contigs: %s", list(var_summary_contigs))

        if missing_contigs:
            logger.info(
                "FAILED contig check, the following contigs are missing: %s",
                missing_contigs,
            )
        if unexpected_contigs:
            logger.info(
                "FAILED contig check, the following contigs are unexpected: %s",
                unexpected_contigs,
            )

    return var_summary


def check_raw_and_adj_callstats(
    t: Union[hl.MatrixTable, hl.Table],
    subsets: List[str],
    verbose: bool,
    delimiter: str = "-",
    metric_first_field: bool = True,
) -> None:
    """
    Perform validity checks on raw and adj data in input Table/MatrixTable.

    Check that:
        - Raw AC and AF are not 0
        - AC and AF are not negative
        - Raw values for AC, AN, nhomalt in each sample subset are greater than or equal to their corresponding adj values

    Raw and adj call stat annotations must be in an info struct annotation on the Table/MatrixTable, e.g. t.info.AC-raw.

    :param t: Input MatrixTable or Table to check.
    :param subsets: List of sample subsets.
    :param verbose: If True, show top values of annotations being checked, including checks that pass; if False, show only top values of annotations that fail checks.
    :param delimiter: String to use as delimiter when making group label combinations. Default is "-".
    :param metric_first_field: If True, metric precedes label group, e.g. AC-afr-male. If False, label group precedes metric, afr-male-AC. Default is True.
    :return: None
    """
    t = t.rows() if isinstance(t, hl.MatrixTable) else t

    field_check_expr = {}

    for group in ["raw", "adj"]:
        # Check AC and nhomalt missing if AN is missing and defined if AN is defined.
        for subfield in ["AC", "nhomalt"]:
            check_field = f"{subfield}{delimiter}{group}"
            an_field = f"AN{delimiter}{group}"
            field_check_expr[
                f"{check_field} defined when AN defined and missing when AN missing"
            ] = {
                "expr": hl.if_else(
                    hl.is_missing(t.info[an_field]),
                    hl.is_defined(t.info[check_field]),
                    hl.is_missing(t.info[check_field]),
                ),
                "agg_func": hl.agg.count_where,
                "display_fields": hl.struct(
                    **{an_field: t.info[an_field], check_field: t.info[check_field]}
                ),
            }

        # Check AF missing if AN is missing and defined if AN is defined and > 0.
        check_field = f"AF{delimiter}{group}"
        an_field = f"AN{delimiter}{group}"
        field_check_expr[
            f"{check_field} defined when AN defined (and > 0) and missing when AN missing"
        ] = {
            "expr": hl.if_else(
                hl.is_missing(t.info[an_field]),
                hl.is_defined(t.info[check_field]),
                (t.info[an_field] > 0) & hl.is_missing(t.info[check_field]),
            ),
            "agg_func": hl.agg.count_where,
            "display_fields": hl.struct(
                **{an_field: t.info[an_field], check_field: t.info[check_field]}
            ),
        }

        # Check raw and adj AF missing if AN is 0.
        check_field = f"AF{delimiter}{group}"
        an_field = f"AN{delimiter}{group}"
        field_check_expr[f"{check_field} missing when AN 0"] = {
            "expr": (t.info[an_field] == 0) & hl.is_defined(t.info[check_field]),
            "agg_func": hl.agg.count_where,
            "display_fields": hl.struct(
                **{an_field: t.info[an_field], check_field: t.info[check_field]}
            ),
        }

    for subfield in ["AC", "AF"]:
        # Check raw AC, AF > 0
        check_field = f"{subfield}{delimiter}raw"
        field_check_expr[f"{check_field} > 0"] = {
            "expr": t.info[check_field] <= 0,
            "agg_func": hl.agg.count_where,
            "display_fields": hl.struct(**{check_field: t.info[check_field]}),
        }

        # Check adj AC, AF > 0
        check_field = f"{subfield}{delimiter}adj"
        field_check_expr[f"{check_field} >= 0"] = {
            "expr": t.info[check_field] < 0,
            "agg_func": hl.agg.count_where,
            "display_fields": hl.struct(
                **{check_field: t.info[check_field], "filters": t.filters}
            ),
        }

    # Check overall gnomad's raw subfields >= adj
    for subfield in ["AC", "AN", "nhomalt"]:
        check_field_left = f"{subfield}{delimiter}raw"
        check_field_right = f"{subfield}{delimiter}adj"

        field_check_expr[f"{check_field_left} >= {check_field_right}"] = {
            "expr": t.info[check_field_left] < t.info[check_field_right],
            "agg_func": hl.agg.count_where,
            "display_fields": hl.struct(
                **{
                    check_field_left: t.info[check_field_left],
                    check_field_right: t.info[check_field_right],
                }
            ),
        }

        for subset in subsets:
            # Add delimiter for subsets but not "" representing entire callset
            if subset:
                subset += delimiter
            field_check_label = (
                f"{subfield}{delimiter}{subset}"
                if metric_first_field
                else f"{subset}{subfield}{delimiter}"
            )
            check_field_left = f"{field_check_label}raw"
            check_field_right = f"{field_check_label}adj"

            field_check_expr[f"{check_field_left} >= {check_field_right}"] = {
                "expr": t.info[check_field_left] < t.info[check_field_right],
                "agg_func": hl.agg.count_where,
                "display_fields": hl.struct(
                    **{
                        check_field_left: t.info[check_field_left],
                        check_field_right: t.info[check_field_right],
                    }
                ),
            }

    generic_field_check_loop(t, field_check_expr, verbose)


def check_sex_chr_metrics(
    t: Union[hl.MatrixTable, hl.Table],
    info_metrics: List[str],
    contigs: List[str],
    verbose: bool,
    delimiter: str = "-",
) -> None:
    """
    Perform validity checks for annotations on the sex chromosomes.

    Check:
        - That metrics for chrY variants in XX samples are NA and not 0
        - That nhomalt counts are equal to XX nhomalt counts for all non-PAR chrX variants

    :param t: Input MatrixTable or Table.
    :param info_metrics: List of metrics in info struct of input Table.
    :param contigs: List of contigs present in input Table.
    :param verbose: If True, show top values of annotations being checked, including checks that pass; if False, show only top values of annotations that fail checks.
    :param delimiter: String to use as the delimiter in XX metrics. Default is "-".
    :return: None
    """
    t = t.rows() if isinstance(t, hl.MatrixTable) else t

    xx_metrics = [x for x in info_metrics if f"{delimiter}XX" in x]

    if "chrY" in contigs:
        logger.info("Check values of XX metrics for Y variants are NA:")
        t_y = hl.filter_intervals(t, [hl.parse_locus_interval("chrY")])
        metrics_values = {}
        for metric in xx_metrics:
            metrics_values[metric] = hl.agg.any(hl.is_defined(t_y.info[metric]))
        output = dict(t_y.aggregate(hl.struct(**metrics_values)))
        for metric, value in output.items():
            if value:
                values_found = t_y.aggregate(
                    hl.agg.filter(
                        hl.is_defined(t_y.info[metric]),
                        hl.agg.take(t_y.info[metric], 1),
                    )
                )
                logger.info(
                    "FAILED %s = %s check for Y variants. Values found: %s",
                    metric,
                    None,
                    values_found,
                )
            else:
                logger.info("PASSED %s = %s check for Y variants", metric, None)

    t_x = hl.filter_intervals(t, [hl.parse_locus_interval("chrX")])
    t_xnonpar = t_x.filter(t_x.locus.in_x_nonpar())
    n = t_xnonpar.count()
    logger.info("Found %d X nonpar sites", n)

    logger.info("Check (nhomalt == nhomalt_xx) for X nonpar variants:")
    xx_metrics = [x for x in xx_metrics if "nhomalt" in x]

    field_check_expr = {}
    for metric in xx_metrics:
        standard_field = metric.replace(f"{delimiter}XX", "")
        check_field_left = f"{metric}"
        check_field_right = f"{standard_field}"
        field_check_expr[f"{check_field_left} == {check_field_right}"] = {
            "expr": t_xnonpar.info[check_field_left]
            != t_xnonpar.info[check_field_right],
            "agg_func": hl.agg.count_where,
            "display_fields": hl.struct(
                **{
                    check_field_left: t_xnonpar.info[check_field_left],
                    check_field_right: t_xnonpar.info[check_field_right],
                }
            ),
        }

    generic_field_check_loop(t_xnonpar, field_check_expr, verbose)


def compute_missingness(
    t: Union[hl.MatrixTable, hl.Table],
    info_metrics: List[str],
    non_info_metrics: List[str],
    n_sites: int,
    missingness_threshold: float,
) -> None:
    """
    Check amount of missingness in all row annotations.

    Print metric to sdout if the percentage of metric annotations missingness exceeds the missingness_threshold.

    :param t: Input MatrixTable or Table.
    :param info_metrics: List of metrics in info struct of input Table.
    :param non_info_metrics: List of row annotations minus info struct from input Table.
    :param n_sites: Number of sites in input Table.
    :param missingness_threshold: Upper cutoff for allowed amount of missingness.
    :return: None
    """
    t = t.rows() if isinstance(t, hl.MatrixTable) else t

    logger.info(
        "Missingness threshold (upper cutoff for what is allowed for missingness"
        " checks): %.2f",
        missingness_threshold,
    )
    metrics_missing = {}
    for x in info_metrics:
        metrics_missing[x] = hl.agg.sum(hl.is_missing(t.info[x]))
    for x in non_info_metrics:
        metrics_missing[x] = hl.agg.sum(hl.is_missing(t[x]))
    output = dict(t.aggregate(hl.struct(**metrics_missing)))

    n_fail = 0
    for metric, n_missing in output.items():
        if n_missing / n_sites > missingness_threshold:
            logger.info(
                "FAILED missingness check for %s: %d sites or %.2f%% missing",
                metric,
                n_missing,
                (100 * n_missing / n_sites),
            )
            n_fail += 1
        else:
            logger.info(
                "Passed missingness check for %s: %d sites or %.2f%% missing",
                metric,
                n_missing,
                (100 * n_missing / n_sites),
            )
    logger.info("%d missing metrics checks failed", n_fail)


def vcf_field_check(
    t: Union[hl.MatrixTable, hl.Table],
    header_dict: Dict[str, Dict[str, Dict[str, str]]],
    row_annotations: List[str] = None,
    entry_annotations: List[str] = None,
    hists: List[str] = HISTS,
) -> bool:
    """
    Check that all VCF fields and descriptions are present in input Table and VCF header dictionary.

    :param t: Input MatrixTable or Table to be exported to VCF.
    :param header_dict: VCF header dictionary.
    :param row_annotations: List of row annotations in MatrixTable or Table.
    :param entry_annotations: List of entry annotations to use if running this check on a MatrixTable.
    :param hists: List of variant histogram annotations. Default is HISTS.
    :return: Boolean with whether all expected fields and descriptions are present.
    """
    hist_fields = []
    for hist in hists:
        hist_fields.append(f"{hist}_bin_freq")
        if "dp" in hist:
            hist_fields.append(f"{hist}_n_larger")

    missing_fields = []
    missing_descriptions = []
    items = ["info", "filter"]
    if entry_annotations:
        items.append("format")
    for item in items:
        if item == "info":
            annots = row_annotations
        elif item == "format":
            annots = entry_annotations
        else:
            annot_t = (
                t.explode_rows(t.filters)
                if isinstance(t, hl.MatrixTable)
                else t.explode(t.filters)
            )
            annots = (
                list(annot_t.aggregate_rows(hl.agg.collect_as_set(annot_t.filters)))
                if isinstance(t, hl.MatrixTable)
                else list(annot_t.aggregate(hl.agg.collect_as_set(annot_t.filters)))
            )

        temp_missing_fields = []
        temp_missing_descriptions = []
        for field in annots:
            try:
                description = header_dict[item][field]
                if len(description) == 0:
                    logger.warning(
                        "%s in info field has empty description in VCF header!", field
                    )
                    temp_missing_descriptions.append(field)
            except KeyError:
                logger.warning("%s in info field does not exist in VCF header!", field)
                # NOTE: END entry is not exported (removed during densify)
                if isinstance(t, hl.MatrixTable) and (field != "END"):
                    temp_missing_fields.append(field)

        missing_fields.extend(temp_missing_fields)
        missing_descriptions.extend(temp_missing_descriptions)

    if len(missing_fields) != 0 or len(missing_descriptions) != 0:
        logger.error(
            "Some fields are either missing or missing descriptions in the VCF header!"
            " Please reconcile."
        )
        logger.error("Missing fields: %s", missing_fields)
        logger.error("Missing descriptions: %s", missing_descriptions)
        return False

    logger.info("Passed VCF fields check!")
    return True


def check_global_and_row_annot_lengths(
    t: Union[hl.MatrixTable, hl.Table],
    row_to_globals_check: Dict[str, List[str]],
    check_all_rows: bool = False,
) -> None:
    """
    Check that the lengths of row annotations match the lengths of associated global annotations.

    :param t: Input MatrixTable or Table.
    :param row_to_globals_check: Dictionary with row annotation (key) and list of associated global annotations (value) to compare.
    :param check_all_rows: If True, check all rows in `t`; if False, check only the first row. Default is False.
    :return: None
    """
    t = t.rows() if isinstance(t, hl.MatrixTable) else t
    if not check_all_rows:
        t = t.head(1)
    for row_field, global_fields in row_to_globals_check.items():
        if not check_all_rows:
            logger.info(
                "Checking length of %s in first row against length of globals: %s",
                row_field,
                global_fields,
            )
        for global_field in global_fields:
            global_len = hl.eval(hl.len(t[global_field]))
            row_len_expr = hl.len(t[row_field])
            failed_rows = t.aggregate(
                hl.struct(
                    n_fail=hl.agg.count_where(row_len_expr != global_len),
                    row_len=hl.agg.counter(row_len_expr),
                )
            )
            outcome = "Failed" if failed_rows["n_fail"] > 0 else "Passed"
            n_rows = t.count()
            logger.info(
                "%s global and row lengths comparison: Length of %s in"
                " globals (%d) does %smatch length of %s in %d out of %d rows (%s)",
                outcome,
                global_field,
                global_len,
                "NOT " if outcome == "Failed" else "",
                row_field,
                failed_rows["n_fail"] if outcome == "Failed" else n_rows,
                n_rows,
                failed_rows["row_len"],
            )


def pprint_global_anns(t: Union[hl.MatrixTable, hl.Table]) -> None:
    """
    Pretty print global annotations.

    :param t: Input MatrixTable or Table.
    """
    global_pprint = {g: hl.eval(t[g]) for g in t.globals}
    pprint(global_pprint, sort_dicts=False)


def validate_release_t(
    t: Union[hl.MatrixTable, hl.Table],
    subsets: List[str] = [""],
    pops: List[str] = POPS[CURRENT_MAJOR_RELEASE]["exomes"],
    missingness_threshold: float = 0.5,
    site_gt_check_expr: Dict[str, hl.expr.BooleanExpression] = None,
    verbose: bool = False,
    show_percent_sites: bool = True,
    delimiter: str = "-",
    metric_first_field: bool = True,
    sum_metrics: List[str] = ["AC", "AN", "nhomalt"],
    sexes: List[str] = SEXES,
    groups: List[str] = ["adj"],
    sample_sum_sets_and_pops: Dict[str, List[str]] = None,
    sort_order: List[str] = SORT_ORDER,
    variant_filter_field: str = "RF",
    problematic_regions: List[str] = ["lcr", "segdup", "nonpar"],
    single_filter_count: bool = False,
    summarize_variants_check: bool = True,
    filters_check: bool = True,
    raw_adj_check: bool = True,
    subset_freq_check: bool = True,
    samples_sum_check: bool = True,
    sex_chr_check: bool = True,
    missingness_check: bool = True,
    pprint_globals: bool = False,
    row_to_globals_check: Optional[Dict[str, List[str]]] = None,
    check_all_rows_in_row_to_global_check: bool = False,
) -> None:
    """
    Perform a battery of validity checks on a specified group of subsets in a MatrixTable containing variant annotations.

    Includes:
    - Summaries of % filter status for different partitions of variants
    - Histogram outlier bin checks
    - Checks on AC, AN, and AF annotations
    - Checks that subgroup annotation values add up to the supergroup annotation values
    - Checks on sex-chromosome annotations; and summaries of % missingness in variant annotations

    All annotations must be within an info struct, e.g. t.info.AC-raw.

    :param t: Input MatrixTable or Table containing variant annotations to check.
    :param subsets: List of subsets to be checked.
    :param pops: List of pops within main callset. Default is POPS[CURRENT_MAJOR_RELEASE]["exomes"].
    :param missingness_threshold: Upper cutoff for allowed amount of missingness. Default is 0.5.
    :param site_gt_check_expr: Optional boolean expression or dictionary of strings and boolean expressions typically used to log how many monoallelic or 100% heterozygous sites are in the Table.
    :param verbose: If True, display top values of relevant annotations being checked, regardless of whether check conditions are violated; if False, display only top values of relevant annotations if check conditions are violated.
    :param show_percent_sites: Show percentage of sites that fail checks. Default is False.
    :param delimiter: String to use as delimiter when making group label combinations. Default is "-".
    :param metric_first_field: If True, metric precedes label group, e.g. AC-afr-male. If False, label group precedes metric, afr-male-AC. Default is True.
    :param sum_metrics: List of metrics to sum and compare to annotationed versions and between subsets and entire callset. Default is ["AC", "AN", "nhomalt"].
    :param sexes: List of sexes in table. Default is SEXES.
    :param groups: List of callstat groups, e.g. "adj" and "raw" contained within the callset. gnomAD does not store the raw callstats for the pop or sex groupings of any subset. Default is ["adj"]
    :param sample_sum_sets_and_pops: Dict with subset (keys) and populations within subset (values) for sample sum check.
    :param sort_order: List containing order to sort label group combinations. Default is SORT_ORDER.
    :param variant_filter_field: String of variant filtration used in the filters annotation on `ht` (e.g. RF, VQSR, AS_VQSR). Default is "RF".
    :param problematic_regions: List of regions considered problematic to run filter check in. Default is ["lcr", "segdup", "nonpar"].
    :param single_filter_count: If True, explode the Table's filter column and give a supplement total count of each filter. Default is False.
    :param summarize_variants_check: When true, runs the summarize_variants method. Default is True.
    :param filters_check: When True, runs the summarize_variant_filters method. Default is True.
    :param raw_adj_check: When True, runs the check_raw_and_adj_callstats method. Default is True.
    :param subset_freq_check: When True, runs the compare_subset_freqs method. Default is True.
    :param samples_sum_check: When True, runs the sum_group_callstats method. Default is True.
    :param sex_chr_check: When True, runs the check_sex_chr_metricss method. Default is True.
    :param missingness_check: When True, runs the compute_missingness method. Default is True.
    :param pprint_globals: When True, Pretty Print the globals of the input Table. Default is True.
    :param row_to_globals_check: Optional dictionary of globals (keys) and rows (values) to be checked. When passed, function checks that the lengths of the global and row annotations are equal.
    :param check_all_rows_in_row_to_global_check: If True, check all rows in `t` in `row_to_globals_check`; if False, check only the first row. Default is False.
    :return: None (stdout display of results from the battery of validity checks).
    """
    if pprint_globals:
        logger.info("GLOBALS OF INPUT TABLE:")
        pprint_global_anns(t)

    if row_to_globals_check is not None:
        logger.info("COMPARE GLOBAL ANNOTATIONS' LENGTHS TO ROW ANNOTATIONS:")
        check_global_and_row_annot_lengths(
            t, row_to_globals_check, check_all_rows_in_row_to_global_check
        )

    if summarize_variants_check:
        logger.info("BASIC SUMMARY OF INPUT TABLE:")
        summarize_variants(t)

    if filters_check:
        logger.info("VARIANT FILTER SUMMARIES:")
        summarize_variant_filters(
            t,
            variant_filter_field,
            problematic_regions,
            single_filter_count,
            site_gt_check_expr,
        )

    if raw_adj_check:
        logger.info("RAW AND ADJ CHECKS:")
        check_raw_and_adj_callstats(t, subsets, verbose, delimiter, metric_first_field)

    if subset_freq_check:
        logger.info("SUBSET FREQUENCY CHECKS:")
        compare_subset_freqs(
            t,
            subsets,
            verbose,
            show_percent_sites,
            delimiter,
            metric_first_field,
            sum_metrics,
        )

    if samples_sum_check:
        logger.info("CALLSET ANNOTATIONS TO SUM GROUP CHECKS:")
        sum_group_callstats(
            t,
            sexes,
            subsets,
            pops,
            groups,
            sample_sum_sets_and_pops,
            verbose,
            sort_order,
            delimiter,
            metric_first_field,
            sum_metrics,
        )

    info_metrics = list(t.row.info)

    if sex_chr_check:
        logger.info("SEX CHROMOSOME ANNOTATION CHECKS:")
        contigs = t.aggregate(hl.agg.collect_as_set(t.locus.contig))
        check_sex_chr_metrics(t, info_metrics, contigs, verbose, delimiter)

    if missingness_check:
        logger.info("MISSINGNESS CHECKS:")
        non_info_metrics = list(t.row)
        non_info_metrics.remove("info")
        n_sites = t.count()
        compute_missingness(
            t, info_metrics, non_info_metrics, n_sites, missingness_threshold
        )
    logger.info("VALIDITY CHECKS COMPLETE")


def count_vep_annotated_variants_per_interval(
    vep_ht: hl.Table, interval_ht: hl.Table
) -> hl.Table:
    """
    Calculate the count of VEP annotated variants in `vep_ht` per interval defined by `interval_ht`.

    .. note::

        - `vep_ht` must contain the 'vep.transcript_consequences' array field, which
          contains a 'biotype' field to determine whether a variant is in a
          "protein-coding" gene.
        - `interval_ht` should be indexed by 'locus' and contain a 'gene_stable_ID'
          field. For example, an interval Table containing the intervals of
          protein-coding genes of a specific Ensembl release.

    The returned Table will have the following fields added:
        - n_total_variants: The number of total variants in the interval.
        - n_pcg_variants: The number of variants in the interval that are annotated as
          "protein-coding".

    :param vep_ht: VEP-annotated Table.
    :param interval_ht: Interval Table.
    :return: Interval Table with annotations for the counts of total variants and
        variants annotated as "protein-coding" in biotype.
    """
    logger.info(
        "Counting the number of total variants and protein-coding variants in each"
        " interval..."
    )

    # Select the vep_ht and annotate genes that have a matched interval from
    # the interval_ht and are protein-coding.
    vep_ht = vep_ht.select(
        gene_stable_ID=interval_ht.index(vep_ht.locus, all_matches=True).gene_stable_ID,
        in_pcg=vep_ht.vep.transcript_consequences.biotype.contains("protein_coding"),
    )

    vep_ht = vep_ht.filter(hl.is_defined(vep_ht.gene_stable_ID))

    # Explode the vep_ht by gene_stable_ID.
    vep_ht = vep_ht.explode(vep_ht.gene_stable_ID)

    # Count the number of total variants and "protein-coding" variants in each interval.
    count_ht = vep_ht.group_by(vep_ht.gene_stable_ID).aggregate(
        all_variants=hl.agg.count(),
        variants_in_pcg=hl.agg.count_where(vep_ht.in_pcg),
    )

    interval_ht = interval_ht.annotate(**count_ht[interval_ht.gene_stable_ID])

    logger.info("Checkpointing the counts per interval...")
    interval_ht = interval_ht.checkpoint(
        new_temp_file("validity_checks.vep_count_per_interval", extension="ht"),
        overwrite=True,
    )

    logger.info("Genes without variants annotated: ")
    gene_sets = interval_ht.aggregate(
        hl.struct(
            na_genes=hl.agg.filter(
                hl.is_missing(interval_ht.variants_in_pcg)
                | (interval_ht.variants_in_pcg == 0),
                hl.agg.collect_as_set(interval_ht.gene_stable_ID),
            ),
            partial_pcg_genes=hl.agg.filter(
                (interval_ht.all_variants != 0)
                & (interval_ht.variants_in_pcg != 0)
                & (interval_ht.all_variants != interval_ht.variants_in_pcg),
                hl.agg.collect_as_set(interval_ht.gene_stable_ID),
            ),
        )
    )

    logger.info(
        "%s gene(s) have no variants annotated as protein-coding in Biotype. It is"
        " likely these genes are not covered by the variants in 'vep_ht'. These"
        " genes are: %s",
        len(gene_sets.na_genes),
        gene_sets.na_genes,
    )

    logger.info(
        "%s gene(s) have a subset of variants annotated as protein-coding biotype"
        " in their defined intervals",
        len(gene_sets.partial_pcg_genes),
    )

    return interval_ht


def check_missingness_of_struct(
    struct_expr: hl.expr.StructExpression, prefix: str = ""
) -> Dict[str, Any]:
    """
    Recursively check the fraction of missing values of all fields within a StructExpression.

    Either a standalone or nested struct can be provided. If the struct contains an array (or set) of values, the array
    will be considered missing if it is NA, an empty array, or only has missing elements.

    :param struct_expr: StructExpression for which to check for missing values.
    :param prefix: Prefix to append to names of struct fields within the struct_expr.
    :return: Dictionary mapping field names to their missingness fraction expressions, with nested dictionaries representing any nested structs.
    """
    if isinstance(struct_expr, hl.expr.StructExpression):
        return {
            f"{prefix}.{key}": check_missingness_of_struct(
                struct_expr[key], f"{prefix}.{key}"
            )
            for key in struct_expr.keys()
        }
    elif isinstance(struct_expr, (hl.expr.ArrayExpression, hl.expr.SetExpression)):
        # Count array/set as missing if it is NA, an empty array/set, or only has missing
        # elements.
        return hl.agg.fraction(
            hl.or_else(struct_expr.all(lambda x: hl.is_missing(x)), True)
        )
    else:
        return hl.agg.fraction(hl.is_missing(struct_expr))


def flatten_missingness_struct(
    missingness_struct: hl.expr.StructExpression,
) -> Dict[str, float]:
    """
    Recursively flatten and evaluate nested dictionaries of missingness within a Struct.

    :param missingness_struct: Struct containing dictionaries of missingness values.
    :return: Dictionary with field names as keys and their evaluated missingness fractions as values.
    """
    missingness_dict = {}
    for key, value in missingness_struct.items():
        # Recursively check nested missingness dictionaries and flatten if needed.
        if isinstance(value, dict):
            missingness_dict.update(flatten_missingness_struct(value))
        else:
            missingness_dict[key] = hl.eval(value)
    return missingness_dict


def unfurl_array_annotations(
    ht: hl.Table, indexed_array_annotations: Dict[str, str]
) -> Dict[str, Any]:
    """
    Unfurl specified arrays of structs into a dictionary of flattened expressions.

    Array annotations must have a corresponding dictionary to define the indices for each array field.
    Example: indexed_array_annotations = {"freq": "freq_index_dict"}, where 'freq' is structured as array<struct{AC: int32, AF: float64, AN: int32, homozygote_count: int64} and 'freq_index_dict' is defined as {'adj': 0, 'raw': 1}.
    The keys of indexed_array_annotations should be present in the Table as row annotations, whereas the values should be present as global annotations.

    :param ht: Input Table.
    :param indexed_array_annotations: Dictionary mapping array field names to their corresponding index dictionaries, which define the indices for each array field. Default is {'faf': 'faf_index_dict', 'freq': 'freq_index_dict'}.
    :return: Flattened dictionary of unfurled array annotations.
    """
    expr_dict = {}

    # For each specified array, unfurl the array elements and their structs
    # into expr_dict.
    for array, array_index_dict in indexed_array_annotations.items():
        # Check for presence of array in the Table rows and the array index in the
        # globals.
        if array not in ht.row:
            raise ValueError(f"Annotation '{array}' not found in the Table rows.")
        if array_index_dict not in ht.globals:
            raise ValueError(
                f"Annotation '{array_index_dict}' not found in the Table globals."
            )

        # Evaluate the index dictionary for the specified array.
        array_index_dict = hl.eval(ht[array_index_dict])

        # Unfurl the array elements and structs into the expression dictionary.
        for k, i in array_index_dict.items():
            for f in ht[array][0].keys():
                expr_dict[f"{f}_{k}"] = ht[array][i][f]

    return expr_dict


def check_array_struct_missingness(
    ht: hl.Table,
    indexed_array_annotations: Dict[str, str] = {
        "faf": "faf_index_dict",
        "freq": "freq_index_dict",
    },
) -> hl.expr.StructExpression:
    """
    Check the missingness of all fields in an array of structs.

    Iterates over arrays of structs and calculates the percentage of missing values for each element of the array and each struct. Array annotations must have a corresponding dictionary to define the indices for each array field.
    Example: indexed_array_annotations = {"freq": "freq_index_dict"}, where 'freq' is structured as array<struct{AC: int32, AF: float64, AN: int32, homozygote_count: int64} and 'freq_index_dict' is defined as {'adj': 0, 'raw': 1}.

    :param ht: Input Table.
    :param indexed_array_annotations: A dictionary mapping array field names to their corresponding index dictionaries, which define the indices for each array field. Default is {'faf': 'faf_index_dict', 'freq': 'freq_index_dict'}.
    :return: A Struct where each field represents a struct field's missingness percentage across the Table for each element of the specified arrays.
    """
    # Create row annotations for each element of the arrays and their structs.
    annotations = unfurl_array_annotations(ht, indexed_array_annotations)

    # Compute missingness for each of the newly created row annotations.
    missingness_dict = {
        field_name: hl.agg.fraction(hl.is_missing(ht[field_name]))
        for field_name in annotations.keys()
    }
    return ht.aggregate(hl.struct(**missingness_dict))


def compute_and_check_summations(
    ht: hl.Table, comparison_groups: Dict[str, Dict[str, Union[List[str], str]]]
) -> Dict[str, int]:
    """
    Compute the number of rows for each specified group where the sum of the specified fields does not match the expected total.

    Example format of comparision_groups:
        {'AC_group_adj_gen_anc': {'values_to_sum': ['AC_afr_adj',
               'AC_amr_adj',
               'AC_asj_adj',
               'AC_eas_adj',
               'AC_fin_adj',
               'AC_mid_adj',
               'AC_nfe_adj',
               'AC_remaining_adj',
               'AC_sas_adj'],
              'expected_total': 'AC_adj'},
              AN_group_adj_gen_anc_sex': {'values_to_sum': ['AN_afr_XX_adj',
               'AN_afr_XY_adj',
               'AN_amr_XX_adj',
               'AN_amr_XY_adj',
               'AN_asj_XX_adj',
               'AN_asj_XY_adj',
               'AN_eas_XX_adj',
               'AN_eas_XY_adj',
               'AN_fin_XX_adj',
               'AN_fin_XY_adj',
               'AN_mid_XX_adj',
               'AN_mid_XY_adj',
               'AN_nfe_XX_adj',
               'AN_nfe_XY_adj',
               'AN_remaining_XX_adj',
               'AN_remaining_XY_adj',
               'AN_sas_XX_adj',
               'AN_sas_XY_adj'],
              'expected_total': 'AN_adj'}}

    :param ht: Table with fields to sum and compare.
    :param comparison_groups: Dictionary describing the groups to sum. Keys are the annotation names to use for the summed totals.
        Values are a dictionary with the 'values_to_sum' key containing a list of fields to sum as values and the 'expected_total'
        key containing the annotation in the Table to which the specified sums should equal.
    :return: Dictionary where keys are group names, and values are the number of rows where the computed sum does not match the expected total.
    """
    # For each group, compute the sum of the fields within 'values_to_sum.'
    summations = {
        group_name: sum(
            ht[field] for field in group_info["values_to_sum"] if field in ht.row
        )
        for group_name, group_info in comparison_groups.items()
    }

    # Annotate the computed sums onto the Table.
    ht = ht.annotate(**summations)

    # Create aggregation expressions to check where the summed values do
    # not equal the expected counts.
    agg_exprs = {
        group_name: hl.agg.count_where(
            ht[group_name] != ht[group_info["expected_total"]]
        )
        for group_name, group_info in comparison_groups.items()
    }

    mismatched_counts = ht.aggregate(hl.struct(**agg_exprs))
    return mismatched_counts

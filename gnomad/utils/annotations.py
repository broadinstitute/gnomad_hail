# noqa: D100

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import hail as hl

import gnomad.utils.filtering as filter_utils
from gnomad.utils.gen_stats import to_phred

logging.basicConfig(
    format="%(asctime)s (%(name)s %(lineno)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ANNOTATIONS_HISTS = {
    "FS": (0, 50, 50),  # NOTE: in 2.0.2 release this was on (0,20)
    "InbreedingCoeff": (-0.25, 0.25, 50),
    "MQ": (0, 80, 40),
    "RAW_MQ": (2, 13, 33),
    "MQRankSum": (-15, 15, 60),
    "QD": (0, 40, 40),
    "ReadPosRankSum": (-15, 15, 60),
    "SOR": (0, 10, 50),
    "BaseQRankSum": (-15, 15, 60),
    "ClippingRankSum": (-5, 5, 40),
    "DP": (1, 9, 32),  # NOTE: in 2.0.2 release this was on (0,8)
    "VQSLOD": (-30, 30, 60),  # NOTE: in 2.0.2 release this was on (-20,20)
    "AS_VQSLOD": (-30, 30, 60),
    "rf_tp_probability": (0, 1, 50),
    "pab_max": (0, 1, 50),
}

VRS_CHROM_IDS = {
    "GRCh38": {
        "chr1": "ga4gh:SQ.Ya6Rs7DHhDeg7YaOSg1EoNi3U_nQ9SvO",
        "chr2": "ga4gh:SQ.pnAqCRBrTsUoBghSD1yp_jXWSmlbdh4g",
        "chr3": "ga4gh:SQ.Zu7h9AggXxhTaGVsy7h_EZSChSZGcmgX",
        "chr4": "ga4gh:SQ.HxuclGHh0XCDuF8x6yQrpHUBL7ZntAHc",
        "chr5": "ga4gh:SQ.aUiQCzCPZ2d0csHbMSbh2NzInhonSXwI",
        "chr6": "ga4gh:SQ.0iKlIQk2oZLoeOG9P1riRU6hvL5Ux8TV",
        "chr7": "ga4gh:SQ.F-LrLMe1SRpfUZHkQmvkVKFEGaoDeHul",
        "chr8": "ga4gh:SQ.209Z7zJ-mFypBEWLk4rNC6S_OxY5p7bs",
        "chr9": "ga4gh:SQ.KEO-4XBcm1cxeo_DIQ8_ofqGUkp4iZhI",
        "chr10": "ga4gh:SQ.ss8r_wB0-b9r44TQTMmVTI92884QvBiB",
        "chr11": "ga4gh:SQ.2NkFm8HK88MqeNkCgj78KidCAXgnsfV1",
        "chr12": "ga4gh:SQ.6wlJpONE3oNb4D69ULmEXhqyDZ4vwNfl",
        "chr13": "ga4gh:SQ._0wi-qoDrvram155UmcSC-zA5ZK4fpLT",
        "chr14": "ga4gh:SQ.eK4D2MosgK_ivBkgi6FVPg5UXs1bYESm",
        "chr15": "ga4gh:SQ.AsXvWL1-2i5U_buw6_niVIxD6zTbAuS6",
        "chr16": "ga4gh:SQ.yC_0RBj3fgBlvgyAuycbzdubtLxq-rE0",
        "chr17": "ga4gh:SQ.dLZ15tNO1Ur0IcGjwc3Sdi_0A6Yf4zm7",
        "chr18": "ga4gh:SQ.vWwFhJ5lQDMhh-czg06YtlWqu0lvFAZV",
        "chr19": "ga4gh:SQ.IIB53T8CNeJJdUqzn9V_JnRtQadwWCbl",
        "chr20": "ga4gh:SQ.-A1QmD_MatoqxvgVxBLZTONHz9-c7nQo",
        "chr21": "ga4gh:SQ.5ZUqxCmDDgN4xTRbaSjN8LwgZironmB8",
        "chr22": "ga4gh:SQ.7B7SHsmchAR0dFcDCuSFjJAo7tX87krQ",
        "chrX": "ga4gh:SQ.w0WZEvgJF0zf_P4yyTzjjv9oW1z61HHP",
        "chrY": "ga4gh:SQ.8_liLu1aycC0tPQPFmUaGXJLDs5SbPZ5",
    },
    "GRCh37": {
        "1": "ga4gh:SQ.S_KjnFVz-FE7M0W6yoaUDgYxLPc1jyWU",
        "2": "ga4gh:SQ.9KdcA9ZpY1Cpvxvg8bMSLYDUpsX6GDLO",
        "3": "ga4gh:SQ.VNBualIltAyi2AI_uXcKU7M9XUOuA7MS",
        "4": "ga4gh:SQ.iy7Zfceb5_VGtTQzJ-v5JpPbpeifHD_V",
        "5": "ga4gh:SQ.vbjOdMfHJvTjK_nqvFvpaSKhZillW0SX",
        "6": "ga4gh:SQ.KqaUhJMW3CDjhoVtBetdEKT1n6hM-7Ek",
        "7": "ga4gh:SQ.IW78mgV5Cqf6M24hy52hPjyyo5tCCd86",
        "8": "ga4gh:SQ.tTm7wmhz0G4lpt8wPspcNkAD_qiminj6",
        "9": "ga4gh:SQ.HBckYGQ4wYG9APHLpjoQ9UUe9v7NxExt",
        "10": "ga4gh:SQ.-BOZ8Esn8J88qDwNiSEwUr5425UXdiGX",
        "11": "ga4gh:SQ.XXi2_O1ly-CCOi3HP5TypAw7LtC6niFG",
        "12": "ga4gh:SQ.105bBysLoDFQHhajooTAUyUkNiZ8LJEH",
        "13": "ga4gh:SQ.Ewb9qlgTqN6e_XQiRVYpoUfZJHXeiUfH",
        "14": "ga4gh:SQ.5Ji6FGEKfejK1U6BMScqrdKJK8GqmIGf",
        "15": "ga4gh:SQ.zIMZb3Ft7RdWa5XYq0PxIlezLY2ccCgt",
        "16": "ga4gh:SQ.W6wLoIFOn4G7cjopxPxYNk2lcEqhLQFb",
        "17": "ga4gh:SQ.AjWXsI7AkTK35XW9pgd3UbjpC3MAevlz",
        "18": "ga4gh:SQ.BTj4BDaaHYoPhD3oY2GdwC_l0uqZ92UD",
        "19": "ga4gh:SQ.ItRDD47aMoioDCNW_occY5fWKZBKlxCX",
        "20": "ga4gh:SQ.iy_UbUrvECxFRX5LPTH_KPojdlT7BKsf",
        "21": "ga4gh:SQ.LpTaNW-hwuY_yARP0rtarCnpCQLkgVCg",
        "22": "ga4gh:SQ.XOgHwwR3Upfp5sZYk6ZKzvV25a4RBVu8",
        "X": "ga4gh:SQ.v7noePfnNpK8ghYXEqZ9NukMXW7YeNsm",
        "Y": "ga4gh:SQ.BT7QyW5iXaX_1PSX-msSGYsqRdMKqkj-",
    },
}


def annotate_with_ht(
    t: Union[hl.MatrixTable, hl.Table],
    annotation_ht: hl.Table,
    fields: Optional[List] = None,
    annotate_cols: bool = False,
    filter_missing: bool = False,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Annotate a MatrixTable/Table with additional annotations from another Table.

    :param t: MatrixTable/Table to be annotated.
    :param annotation_ht: Table containing additional annotations to be joined on `t`.
    :param fields: Optional list of fields to select from `annotation_ht` and add to `t`
    :param annotate_cols: If True, annotate columns instead of rows. Default is False.
    :param filter_missing: If True, filter out missing rows/cols in `t` that are not
        present in `annotation_ht`. Default is False.
    :return: Annotated MatrixTable/Table.
    """
    if fields is not None:
        annotation_ht = annotation_ht.select(*fields)

    if filter_missing:
        logger.info("Filtering input to variants in the supplied annotation HT...")
        if isinstance(t, hl.Table):
            t = t.semi_join(annotation_ht)
        elif annotate_cols:
            t = t.semi_join_cols(annotation_ht)
        else:
            t = t.semi_join_rows(annotation_ht)

    if isinstance(t, hl.Table):
        t = t.annotate(**annotation_ht[t.key])
    elif annotate_cols:
        t = t.annotate_cols(**annotation_ht[t.col_key])
    else:
        t = t.annotate_rows(**annotation_ht[t.row_key])

    return t


def pop_max_expr(
    freq: hl.expr.ArrayExpression,
    freq_meta: hl.expr.ArrayExpression,
    pops_to_exclude: Optional[Set[str]] = None,
    pop_label: str = "pop",
) -> hl.expr.StructExpression:
    """

    Create an expression containing the frequency information about the population that has the highest AF in `freq_meta`.

    Populations specified in `pops_to_exclude` are excluded and only frequencies from adj populations are considered.

    This resulting struct contains the following fields:

        - AC: int32
        - AF: float64
        - AN: int32
        - homozygote_count: int32
        - pop: str

    :param freq: ArrayExpression of Structs with fields ['AC', 'AF', 'AN', 'homozygote_count']
    :param freq_meta: ArrayExpression of meta dictionaries corresponding to freq (as returned by annotate_freq)
    :param pops_to_exclude: Set of populations to skip for popmax calcluation
    :param pop_label: Label of the population field in the meta dictionary
    :return: Popmax struct
    """
    _pops_to_exclude = (
        hl.literal(pops_to_exclude)
        if pops_to_exclude is not None
        else hl.empty_set(hl.tstr)
    )

    # pylint: disable=invalid-unary-operand-type
    popmax_freq_indices = hl.range(0, hl.len(freq_meta)).filter(
        lambda i: (hl.set(freq_meta[i].keys()) == {"group", pop_label})
        & (freq_meta[i]["group"] == "adj")
        & (~_pops_to_exclude.contains(freq_meta[i][pop_label]))
    )
    freq_filtered = popmax_freq_indices.map(
        lambda i: freq[i].annotate(**{pop_label: freq_meta[i][pop_label]})
    ).filter(lambda f: f.AC > 0)

    sorted_freqs = hl.sorted(freq_filtered, key=lambda x: x.AF, reverse=True)
    return hl.or_missing(hl.len(sorted_freqs) > 0, sorted_freqs[0])


def project_max_expr(
    project_expr: hl.expr.StringExpression,
    gt_expr: hl.expr.CallExpression,
    alleles_expr: hl.expr.ArrayExpression,
    n_projects: int = 5,
) -> hl.expr.ArrayExpression:
    """
    Create an expression that computes allele frequency information by project for the `n_projects` with the largest AF at this row.

    Will return an array with one element per non-reference allele.

    Each of these elements is itself an array of structs with the following fields:

        - AC: int32
        - AF: float64
        - AN: int32
        - homozygote_count: int32
        - project: str

    .. note::

        Only projects with AF > 0 are returned.
        In case of ties, the project ordering is not guaranteed, and at most `n_projects` are returned.

    :param project_expr: column expression containing the project
    :param gt_expr: entry expression containing the genotype
    :param alleles_expr: row expression containing the alleles
    :param n_projects: Maximum number of projects to return for each row
    :return: projectmax expression
    """
    n_alleles = hl.len(alleles_expr)

    # compute call stats by  project
    project_cs = hl.array(
        hl.agg.group_by(project_expr, hl.agg.call_stats(gt_expr, alleles_expr))
    )

    return hl.or_missing(
        n_alleles > 1,  # Exclude monomorphic sites
        hl.range(1, n_alleles).map(
            lambda ai: hl.sorted(
                project_cs.filter(
                    # filter to projects with AF > 0
                    lambda x: x[1].AF[ai]
                    > 0
                ),
                # order the callstats computed by AF in decreasing order
                lambda x: -x[1].AF[ai],
                # take the n_projects projects with largest AF
            )[:n_projects].map(
                # add the project in the callstats struct
                lambda x: x[1].annotate(
                    AC=x[1].AC[ai],
                    AF=x[1].AF[ai],
                    AN=x[1].AN,
                    homozygote_count=x[1].homozygote_count[ai],
                    project=x[0],
                )
            )
        ),
    )


def faf_expr(
    freq: hl.expr.ArrayExpression,
    freq_meta: hl.expr.ArrayExpression,
    locus: hl.expr.LocusExpression,
    pops_to_exclude: Optional[Set[str]] = None,
    faf_thresholds: List[float] = [0.95, 0.99],
    pop_label: str = "pop",
) -> Tuple[hl.expr.ArrayExpression, List[Dict[str, str]]]:
    """
    Calculate the filtering allele frequency (FAF) for each threshold specified in `faf_thresholds`.

    See http://cardiodb.org/allelefrequencyapp/ for more information.

    The FAF is computed for each of the following population stratification if found in `freq_meta`:

        - All samples, with adj criteria
        - For each population, with adj criteria
        - For all sex/population on the non-PAR regions of sex chromosomes (will be missing on autosomes and PAR regions of sex chromosomes)

    Each of the FAF entry is a struct with one entry per threshold specified in `faf_thresholds` of type float64.

    This returns a tuple with two expressions:

        1. An array of FAF expressions as described above
        2. An array of dict containing the metadata for each of the array elements, in the same format as that produced by `annotate_freq`.

    :param freq: ArrayExpression of call stats structs (typically generated by hl.agg.call_stats)
    :param freq_meta: ArrayExpression of meta dictionaries corresponding to freq (typically generated using annotate_freq)
    :param locus: locus
    :param pops_to_exclude: Set of populations to exclude from faf calculation (typically bottlenecked or consanguineous populations)
    :param faf_thresholds: List of FAF thresholds to compute
    :param pop_label: Label of the population field in the meta dictionary
    :return: (FAF expression, FAF metadata)
    """
    _pops_to_exclude = (
        hl.literal(pops_to_exclude)
        if pops_to_exclude is not None
        else hl.empty_set(hl.tstr)
    )

    # pylint: disable=invalid-unary-operand-type
    faf_freq_indices = hl.range(0, hl.len(freq_meta)).filter(
        lambda i: (freq_meta[i].get("group") == "adj")
        & (
            (freq_meta[i].size() == 1)
            | (
                (hl.set(freq_meta[i].keys()) == {pop_label, "group"})
                & (~_pops_to_exclude.contains(freq_meta[i][pop_label]))
            )
        )
    )
    sex_faf_freq_indices = hl.range(0, hl.len(freq_meta)).filter(
        lambda i: (freq_meta[i].get("group") == "adj")
        & (freq_meta[i].contains("sex"))
        & (
            (freq_meta[i].size() == 2)
            | (
                (hl.set(freq_meta[i].keys()) == {pop_label, "group", "sex"})
                & (~_pops_to_exclude.contains(freq_meta[i][pop_label]))
            )
        )
    )

    faf_expr = faf_freq_indices.map(
        lambda i: hl.struct(
            **{
                f"faf{str(threshold)[2:]}": hl.experimental.filtering_allele_frequency(
                    freq[i].AC, freq[i].AN, threshold
                )
                for threshold in faf_thresholds
            }
        )
    )

    faf_expr = faf_expr.extend(
        sex_faf_freq_indices.map(
            lambda i: hl.or_missing(
                ~locus.in_autosome_or_par(),
                hl.struct(
                    **{
                        f"faf{str(threshold)[2:]}": (
                            hl.experimental.filtering_allele_frequency(
                                freq[i].AC, freq[i].AN, threshold
                            )
                        )
                        for threshold in faf_thresholds
                    }
                ),
            )
        )
    )

    faf_meta = faf_freq_indices.extend(sex_faf_freq_indices).map(lambda i: freq_meta[i])
    return faf_expr, hl.eval(faf_meta)


def gen_anc_faf_max_expr(
    faf: hl.expr.ArrayExpression,
    faf_meta: hl.expr.ArrayExpression,
    pop_label: str = "pop",
) -> hl.expr.StructExpression:
    """
    Retrieve the maximum FAF and corresponding genetic ancestry for each of the thresholds in `faf`.

    This resulting struct contains the following fields:

        - faf95_max: float64
        - faf95_max_gen_anc: str
        - faf99_max: float64
        - faf99_max_gen_anc: str

    :param faf: ArrayExpression of Structs of FAF thresholds previously computed. When
        `faf_expr` is used, contains fields 'faf95' and 'faf99'.
    :param faf_meta: ArrayExpression of meta dictionaries corresponding to faf (as
        returned by faf_expr)
    :param pop_label: Label of the population field in the meta dictionary
    :return: Genetic ancestry group struct for FAF max
    """
    faf_gen_anc_indices = hl.enumerate(faf_meta).filter(
        lambda i: (hl.set(i[1].keys()) == {"group", pop_label})
        & (i[1]["group"] == "adj")
    )
    max_fafs_expr = hl.struct()

    # Iterate through faf thresholds, generally 'faf95' and 'faf99', and
    # take the maximum faf value, '[0]', and its gen_anc from the sorted faf array
    for threshold in faf[0].keys():
        faf_struct = hl.sorted(
            faf_gen_anc_indices.map(
                lambda x: {
                    f"{threshold}_max": hl.or_missing(
                        faf[x[0]][threshold] > 0, faf[x[0]][threshold]
                    ),
                    f"{threshold}_max_gen_anc": hl.or_missing(
                        faf[x[0]][threshold] > 0, x[1][pop_label]
                    ),
                }
            ),
            key=lambda faf: faf[f"{threshold}_max"],
            reverse=True,
        )[0]

        max_fafs_expr = max_fafs_expr.annotate(**faf_struct)

    return max_fafs_expr


def qual_hist_expr(
    gt_expr: Optional[hl.expr.CallExpression] = None,
    gq_expr: Optional[hl.expr.NumericExpression] = None,
    dp_expr: Optional[hl.expr.NumericExpression] = None,
    ad_expr: Optional[hl.expr.ArrayNumericExpression] = None,
    adj_expr: Optional[hl.expr.BooleanExpression] = None,
    ab_expr: Optional[hl.expr.NumericExpression] = None,
    split_adj_and_raw: bool = False,
) -> hl.expr.StructExpression:
    """
    Return a struct expression with genotype quality histograms based on the arguments given (dp, gq, ad, ab).

    .. note::

        - If `gt_expr` is provided, will return histograms for non-reference samples only as well as all samples.
        - `gt_expr` is required for the allele-balance histogram, as it is only computed on het samples.
        - If `ab_expr` is provided, the allele-balance histogram is computed using this expression instead of the ad_expr.
        - If `adj_expr` is provided, additional histograms are computed using only adj samples.

    :param gt_expr: Entry expression containing genotype.
    :param gq_expr: Entry expression containing genotype quality.
    :param dp_expr: Entry expression containing depth.
    :param ad_expr: Entry expression containing allelic depth (bi-allelic here).
    :param adj_expr: Entry expression containing adj (high quality) genotype status.
    :param ab_expr: Entry expression containing allele balance (bi-allelic here).
    :param split_adj_and_raw: Whether to split the adj and raw histograms into separate fields in the returned struct expr.
    :return: Genotype quality histograms expression.
    """
    qual_hists = {}
    if gq_expr is not None:
        qual_hists["gq_hist"] = hl.agg.hist(gq_expr, 0, 100, 20)
    if dp_expr is not None:
        qual_hists["dp_hist"] = hl.agg.hist(dp_expr, 0, 100, 20)

    if gt_expr is not None:
        qual_hists = {
            **{
                f"{qual_hist_name}_all": qual_hist_expr
                for qual_hist_name, qual_hist_expr in qual_hists.items()
            },
            **{
                f"{qual_hist_name}_alt": hl.agg.filter(
                    gt_expr.is_non_ref(), qual_hist_expr
                )
                for qual_hist_name, qual_hist_expr in qual_hists.items()
            },
        }
        ab_hist_msg = "Using the %s to compute allele balance histogram..."
        if ab_expr is not None:
            logger.info(ab_hist_msg, "ab_expr")
            qual_hists["ab_hist_alt"] = hl.agg.filter(
                gt_expr.is_het(), hl.agg.hist(ab_expr, 0, 1, 20)
            )
        elif ad_expr is not None:
            logger.info(ab_hist_msg, "ad_expr")
            qual_hists["ab_hist_alt"] = hl.agg.filter(
                gt_expr.is_het(), hl.agg.hist(ad_expr[1] / hl.sum(ad_expr), 0, 1, 20)
            )

    else:
        qual_hists = {
            f"{qual_hist_name}_all": qual_hist_expr
            for qual_hist_name, qual_hist_expr in qual_hists.items()
        }

    if adj_expr is not None:
        adj_qual_hists = {
            qual_hist_name: hl.agg.filter(adj_expr, qual_hist_expr)
            for qual_hist_name, qual_hist_expr in qual_hists.items()
        }
        if split_adj_and_raw:
            return hl.struct(
                raw_qual_hists=hl.struct(**qual_hists),
                qual_hists=hl.struct(**adj_qual_hists),
            )
        else:
            qual_hists.update({f"{k}_adj": v for k, v in adj_qual_hists.items()})
    return hl.struct(**qual_hists)


def age_hists_expr(
    adj_expr: hl.expr.BooleanExpression,
    gt_expr: hl.expr.CallExpression,
    age_expr: hl.expr.NumericExpression,
    lowest_boundary: int = 30,
    highest_boundary: int = 80,
    n_bins: int = 10,
) -> hl.expr.StructExpression:
    """
    Return a StructExpression with the age histograms for hets and homs.

    :param adj_expr: Entry expression containing whether a genotype is high quality (adj) or not
    :param gt_expr: Entry expression containing the genotype
    :param age_expr: Col expression containing the sample's age
    :param lowest_boundary: Lowest bin boundary (any younger sample will be binned in n_smaller)
    :param highest_boundary: Highest bin boundary (any older sample will be binned in n_larger)
    :param n_bins: Total number of bins
    :return: A struct with `age_hist_het` and `age_hist_hom`
    """
    return hl.struct(
        age_hist_het=hl.agg.filter(
            adj_expr & gt_expr.is_het(),
            hl.agg.hist(age_expr, lowest_boundary, highest_boundary, n_bins),
        ),
        age_hist_hom=hl.agg.filter(
            adj_expr & gt_expr.is_hom_var(),
            hl.agg.hist(age_expr, lowest_boundary, highest_boundary, n_bins),
        ),
    )


def get_lowqual_expr(
    alleles: hl.expr.ArrayExpression,
    qual_approx_expr: Union[hl.expr.ArrayNumericExpression, hl.expr.NumericExpression],
    snv_phred_threshold: int = 30,
    snv_phred_het_prior: int = 30,  # 1/1000
    indel_phred_threshold: int = 30,
    indel_phred_het_prior: int = 39,  # 1/8,000
) -> Union[hl.expr.BooleanExpression, hl.expr.ArrayExpression]:
    """
    Compute lowqual threshold expression for either split or unsplit alleles based on QUALapprox or AS_QUALapprox.

    .. note::

        When running This lowqual annotation using QUALapprox, it differs from the GATK LowQual filter.
        This is because GATK computes this annotation at the site level, which uses the least stringent prior for mixed sites.
        When run using AS_QUALapprox, this implementation can thus be more stringent for certain alleles at mixed sites.

    :param alleles: Array of alleles
    :param qual_approx_expr: QUALapprox or AS_QUALapprox
    :param snv_phred_threshold: Phred-scaled SNV "emission" threshold (similar to GATK emission threshold)
    :param snv_phred_het_prior: Phred-scaled SNV heterozygosity prior (30 = 1/1000 bases, GATK default)
    :param indel_phred_threshold: Phred-scaled indel "emission" threshold (similar to GATK emission threshold)
    :param indel_phred_het_prior: Phred-scaled indel heterozygosity prior (30 = 1/1000 bases, GATK default)
    :return: lowqual expression (BooleanExpression if `qual_approx_expr`is Numeric, Array[BooleanExpression] if `qual_approx_expr` is ArrayNumeric)
    """
    min_snv_qual = snv_phred_threshold + snv_phred_het_prior
    min_indel_qual = indel_phred_threshold + indel_phred_het_prior
    min_mixed_qual = max(min_snv_qual, min_indel_qual)

    if isinstance(qual_approx_expr, hl.expr.ArrayNumericExpression):
        return hl.range(1, hl.len(alleles)).map(
            lambda ai: hl.if_else(
                hl.is_snp(alleles[0], alleles[ai]),
                qual_approx_expr[ai - 1] < min_snv_qual,
                qual_approx_expr[ai - 1] < min_indel_qual,
            )
        )
    else:
        return (
            hl.case()
            .when(
                hl.range(1, hl.len(alleles)).all(
                    lambda ai: hl.is_snp(alleles[0], alleles[ai])
                ),
                qual_approx_expr < min_snv_qual,
            )
            .when(
                hl.range(1, hl.len(alleles)).all(
                    lambda ai: hl.is_indel(alleles[0], alleles[ai])
                ),
                qual_approx_expr < min_indel_qual,
            )
            .default(qual_approx_expr < min_mixed_qual)
        )


def get_annotations_hists(
    ht: hl.Table,
    annotations_hists: Dict[str, Tuple],
    log10_annotations: List[str] = ["DP"],
) -> Dict[str, hl.expr.StructExpression]:
    """
    Create histograms for variant metrics in ht.info.

    Used when creating site quality distribution json files.

    :param ht: Table with variant metrics
    :param annotations_hists: Dictionary of metrics names and their histogram values (start, end, bins)
    :param log10_annotations: List of metrics to log scale
    :return: Dictionary of merics and their histograms
    :rtype: Dict[str, hl.expr.StructExpression]
    """
    # Check all fields in ht.info and create histograms if they are in
    # annotations_hists dict
    return {
        field: hl.agg.hist(
            hl.log10(ht.info[field]) if field in log10_annotations else ht.info[field],
            start,
            end,
            bins,
        )
        for field, (start, end, bins) in annotations_hists.items()
        if field in ht.row.info
    }


def create_frequency_bins_expr(
    AC: hl.expr.NumericExpression, AF: hl.expr.NumericExpression
) -> hl.expr.StringExpression:
    """
    Create bins for frequencies in preparation for aggregating QUAL by frequency bin.

    Bins:
        - singleton
        - doubleton
        - 0.00005
        - 0.0001
        - 0.0002
        - 0.0005
        - 0.001,
        - 0.002
        - 0.005
        - 0.01
        - 0.02
        - 0.05
        - 0.1
        - 0.2
        - 0.5
        - 1

    NOTE: Frequencies should be frequencies from raw data.
    Used when creating site quality distribution json files.

    :param AC: Field in input that contains the allele count information
    :param AF: Field in input that contains the allele frequency information
    :return: Expression containing bin name
    :rtype: hl.expr.StringExpression
    """
    bin_expr = (
        hl.case()
        .when(AC == 1, "binned_singleton")
        .when(AC == 2, "binned_doubleton")
        .when((AC > 2) & (AF < 0.00005), "binned_0.00005")
        .when((AF >= 0.00005) & (AF < 0.0001), "binned_0.0001")
        .when((AF >= 0.0001) & (AF < 0.0002), "binned_0.0002")
        .when((AF >= 0.0002) & (AF < 0.0005), "binned_0.0005")
        .when((AF >= 0.0005) & (AF < 0.001), "binned_0.001")
        .when((AF >= 0.001) & (AF < 0.002), "binned_0.002")
        .when((AF >= 0.002) & (AF < 0.005), "binned_0.005")
        .when((AF >= 0.005) & (AF < 0.01), "binned_0.01")
        .when((AF >= 0.01) & (AF < 0.02), "binned_0.02")
        .when((AF >= 0.02) & (AF < 0.05), "binned_0.05")
        .when((AF >= 0.05) & (AF < 0.1), "binned_0.1")
        .when((AF >= 0.1) & (AF < 0.2), "binned_0.2")
        .when((AF >= 0.2) & (AF < 0.5), "binned_0.5")
        .when((AF >= 0.5) & (AF <= 1), "binned_1")
        .default(hl.null(hl.tstr))
    )
    return bin_expr


def annotate_and_index_source_mt_for_sex_ploidy(
    locus_expr: hl.expr.LocusExpression,
    karyotype_expr: hl.expr.StringExpression,
    xy_karyotype_str: str = "XY",
    xx_karyotype_str: str = "XX",
) -> Tuple[hl.expr.StructExpression, hl.expr.StructExpression]:
    """
    Prepare relevant ploidy annotations for downstream calculations on a matrix table.

    This method is used as an optimization for the `get_is_haploid_expr` and
    `adjusted_sex_ploidy_expr` methods.

    This method annotates the `locus_expr` source matrix table with the following
    fields:

        - `xy`: Boolean indicating if the sample is XY.
        - `xx`: Boolean indicating if the sample is XX.
        - `in_non_par`: Boolean indicating if the locus is in a non-PAR region.
        - `x_nonpar`: Boolean indicating if the locus is in a non-PAR region of the X
          chromosome.
        - `y_par`: Boolean indicating if the locus is in a PAR region of the Y
          chromosome.
        - `y_nonpar`: Boolean indicating if the locus is in a non-PAR region of the Y
          chromosome.

    :param locus_expr: Locus expression.
    :param karyotype_expr: Karyotype expression.
    :param xy_karyotype_str: String representing XY karyotype. Default is "XY".
    :param xx_karyotype_str: String representing XX karyotype. Default is "XX".
    :return: Tuple of index expressions for columns and rows.
    """
    source_mt = locus_expr._indices.source
    col_ht = source_mt.annotate_cols(
        xy=karyotype_expr.upper() == xy_karyotype_str,
        xx=karyotype_expr.upper() == xx_karyotype_str,
    ).cols()
    row_ht = source_mt.annotate_rows(
        in_non_par=~locus_expr.in_autosome_or_par(),
        in_autosome=locus_expr.in_autosome(),
        x_nonpar=locus_expr.in_x_nonpar(),
        y_par=locus_expr.in_y_par(),
        y_nonpar=locus_expr.in_y_nonpar(),
    ).rows()
    col_idx = col_ht[source_mt.col_key]
    row_idx = row_ht[source_mt.row_key]

    return col_idx, row_idx


def get_is_haploid_expr(
    gt_expr: Optional[hl.expr.CallExpression] = None,
    locus_expr: Optional[hl.expr.LocusExpression] = None,
    karyotype_expr: Optional[hl.expr.StringExpression] = None,
    xy_karyotype_str: str = "XY",
    xx_karyotype_str: str = "XX",
) -> hl.expr.BooleanExpression:
    """
    Determine if a genotype or locus and karyotype combination is haploid.

    .. note::

        One of `gt_expr` or `locus_expr` and `karyotype_expr` is required.

    :param gt_expr: Optional genotype expression.
    :param locus_expr: Optional locus expression.
    :param karyotype_expr: Optional sex karyotype expression.
    :param xy_karyotype_str: String representing XY karyotype. Default is "XY".
    :param xx_karyotype_str: String representing XX karyotype. Default is "XX".
    :return: Boolean expression indicating if the genotype is haploid.
    """
    if gt_expr is None and locus_expr is None and karyotype_expr is None:
        raise ValueError(
            "One of 'gt_expr' or 'locus_expr' and 'karyotype_expr' is required."
        )

    if gt_expr is not None:
        return gt_expr.is_haploid()

    if locus_expr is None or karyotype_expr is None:
        raise ValueError(
            "Both 'locus_expr' and 'karyotype_expr' are required if no 'gt_expr' is "
            "supplied."
        )
    # An optimization that annotates the locus's matrix table with the
    # fields in the case statements below as an optimization step
    col_idx, row_idx = annotate_and_index_source_mt_for_sex_ploidy(
        locus_expr, karyotype_expr, xy_karyotype_str, xx_karyotype_str
    )

    return row_idx.in_non_par & hl.or_missing(
        ~(col_idx.xx & (row_idx.y_par | row_idx.y_nonpar)),
        col_idx.xy & (row_idx.x_nonpar | row_idx.y_nonpar),
    )


def get_gq_dp_adj_expr(
    gq_expr: Union[hl.expr.Int32Expression, hl.expr.Int64Expression],
    dp_expr: Union[hl.expr.Int32Expression, hl.expr.Int64Expression],
    gt_expr: Optional[hl.expr.CallExpression] = None,
    locus_expr: Optional[hl.expr.LocusExpression] = None,
    karyotype_expr: Optional[hl.expr.StringExpression] = None,
    adj_gq: int = 20,
    adj_dp: int = 10,
    haploid_adj_dp: int = 5,
) -> hl.expr.BooleanExpression:
    """
    Get adj annotation using only GQ and DP.

    Default thresholds correspond to gnomAD values.

    .. note::

        This function can be used to annotate adj taking into account only GQ and DP.
        It is useful for cases where the GT field is not available, such as in the
        reference data of a VariantDataset.

    .. note::

        One of `gt_expr` or `locus_expr` and `karyotype_expr` is required.

    :param gq_expr: GQ expression.
    :param dp_expr: DP expression.
    :param gt_expr: Optional genotype expression.
    :param locus_expr: Optional locus expression.
    :param karyotype_expr: Optional sex karyotype expression.
    :param adj_gq: GQ threshold for adj. Default is 20.
    :param adj_dp: DP threshold for adj. Default is 10.
    :param haploid_adj_dp: Haploid DP threshold for adj. Default is 5.
    :return: Boolean expression indicating adj filter.
    """
    return (gq_expr >= adj_gq) & hl.if_else(
        get_is_haploid_expr(gt_expr, locus_expr, karyotype_expr),
        dp_expr >= haploid_adj_dp,
        dp_expr >= adj_dp,
    )


def get_het_ab_adj_expr(
    gt_expr: hl.expr.CallExpression,
    dp_expr: Union[hl.expr.Int32Expression, hl.expr.Int64Expression],
    ad_expr: hl.expr.ArrayNumericExpression,
    adj_ab: float = 0.2,
) -> hl.expr.BooleanExpression:
    """
    Get adj het AB annotation.

    :param gt_expr: Genotype expression.
    :param dp_expr: DP expression.
    :param ad_expr: AD expression.
    :param adj_ab: AB threshold for adj. Default is 0.2.
    :return: Boolean expression indicating adj het AB filter.
    """
    return (
        hl.case()
        .when(~gt_expr.is_het(), True)
        .when(gt_expr.is_het_ref(), ad_expr[gt_expr[1]] / dp_expr >= adj_ab)
        .default(
            (ad_expr[gt_expr[0]] / dp_expr >= adj_ab)
            & (ad_expr[gt_expr[1]] / dp_expr >= adj_ab)
        )
    )


def get_adj_expr(
    gt_expr: hl.expr.CallExpression,
    gq_expr: Union[hl.expr.Int32Expression, hl.expr.Int64Expression],
    dp_expr: Union[hl.expr.Int32Expression, hl.expr.Int64Expression],
    ad_expr: hl.expr.ArrayNumericExpression,
    adj_gq: int = 20,
    adj_dp: int = 10,
    adj_ab: float = 0.2,
    haploid_adj_dp: int = 5,
) -> hl.expr.BooleanExpression:
    """
    Get adj genotype annotation.

    Defaults correspond to gnomAD values.
    """
    return get_gq_dp_adj_expr(
        gq_expr,
        dp_expr,
        gt_expr=gt_expr,
        adj_gq=adj_gq,
        adj_dp=adj_dp,
        haploid_adj_dp=haploid_adj_dp,
    ) & get_het_ab_adj_expr(gt_expr, dp_expr, ad_expr, adj_ab)


def annotate_adj(
    mt: hl.MatrixTable,
    adj_gq: int = 20,
    adj_dp: int = 10,
    adj_ab: float = 0.2,
    haploid_adj_dp: int = 5,
) -> hl.MatrixTable:
    """
    Annotate genotypes with adj criteria (assumes diploid).

    Defaults correspond to gnomAD values.
    """
    if "GT" not in mt.entry and "LGT" in mt.entry:
        logger.warning("No GT field found, using LGT instead.")
        gt_expr = mt.LGT
    else:
        gt_expr = mt.GT

    if "AD" not in mt.entry and "LAD" in mt.entry:
        logger.warning("No AD field found, using LAD instead.")
        ad_expr = mt.LAD
    else:
        ad_expr = mt.AD

    return mt.annotate_entries(
        adj=get_adj_expr(
            gt_expr, mt.GQ, mt.DP, ad_expr, adj_gq, adj_dp, adj_ab, haploid_adj_dp
        )
    )


def add_variant_type(alt_alleles: hl.expr.ArrayExpression) -> hl.expr.StructExpression:
    """Get Struct of variant_type and n_alt_alleles from ArrayExpression of Strings."""
    ref = alt_alleles[0]
    alts = alt_alleles[1:]
    non_star_alleles = hl.filter(lambda a: a != "*", alts)
    return hl.struct(
        variant_type=hl.if_else(
            hl.all(lambda a: hl.is_snp(ref, a), non_star_alleles),
            hl.if_else(hl.len(non_star_alleles) > 1, "multi-snv", "snv"),
            hl.if_else(
                hl.all(lambda a: hl.is_indel(ref, a), non_star_alleles),
                hl.if_else(hl.len(non_star_alleles) > 1, "multi-indel", "indel"),
                "mixed",
            ),
        ),
        n_alt_alleles=hl.len(non_star_alleles),
    )


def annotate_allele_info(ht: hl.Table) -> hl.Table:
    """
    Return bi-allelic sites Table with an 'allele_info' annotation.

    .. note::

        This function requires that the input `ht` is unsplit and returns a split `ht`.

    'allele_info' is a struct with the following information:
        - variant_type: Variant type (snv, indel, multi-snv, multi-indel, or mixed).
        - n_alt_alleles: Total number of alternate alleles observed at variant locus.
        - has_star: True if the variant contains a star allele.
        - allele_type: Allele type (snv, insertion, deletion, or mixed).
        - was_mixed: True if the variant was mixed (i.e. contained both SNVs and indels).
        - nonsplit_alleles: Array of alleles before splitting.

    :param Table ht: Unsplit input Table.
    :return: Split Table with allele data annotation added,
    """
    ht = ht.annotate(
        allele_info=hl.struct(
            **add_variant_type(ht.alleles),
            has_star=hl.any(lambda a: a == "*", ht.alleles),
        )
    )

    ht = hl.split_multi(ht)

    ref_expr = ht.alleles[0]
    alt_expr = ht.alleles[1]
    allele_type_expr = (
        hl.case()
        .when(hl.is_snp(ref_expr, alt_expr), "snv")
        .when(hl.is_insertion(ref_expr, alt_expr), "ins")
        .when(hl.is_deletion(ref_expr, alt_expr), "del")
        .default("complex")
    )
    ht = ht.transmute(
        allele_info=ht.allele_info.annotate(
            allele_type=allele_type_expr,
            was_mixed=ht.allele_info.variant_type == "mixed",
            nonsplit_alleles=ht.old_alleles,
        )
    )

    return ht


def annotation_type_is_numeric(t: Any) -> bool:
    """
    Given an annotation type, return whether it is a numerical type or not.

    :param t: Type to test
    :return: If the input type is numeric
    """
    return t in (hl.tint32, hl.tint64, hl.tfloat32, hl.tfloat64)


def annotation_type_in_vcf_info(t: Any) -> bool:
    """
    Given an annotation type, returns whether that type can be natively exported to a VCF INFO field.

    .. note::

        Types that aren't natively exportable to VCF will be converted to String on export.

    :param t: Type to test
    :return: If the input type can be exported to VCF
    """
    return (
        annotation_type_is_numeric(t)
        or t in (hl.tstr, hl.tbool)
        or (
            isinstance(t, (hl.tarray, hl.tset))
            and annotation_type_in_vcf_info(t.element_type)
        )
    )


def bi_allelic_site_inbreeding_expr(
    call: Optional[hl.expr.CallExpression] = None,
    callstats_expr: Optional[hl.expr.StructExpression] = None,
) -> hl.expr.Float32Expression:
    """
    Return the site inbreeding coefficient as an expression to be computed on a MatrixTable.

    This is implemented based on the GATK InbreedingCoeff metric:
    https://software.broadinstitute.org/gatk/documentation/article.php?id=8032

    .. note::

        The computation is run based on the counts of alternate alleles and thus should only be run on bi-allelic sites.

    :param call: Expression giving the calls in the MT
    :param callstats_expr: StructExpression containing only alternate allele AC, AN, and homozygote_count as integers. If passed, used to create expression in place of GT calls.
    :return: Site inbreeding coefficient expression
    """
    if call is None and callstats_expr is None:
        raise ValueError("One of `call` or `callstats_expr` must be passed.")

    def inbreeding_coeff(
        gt_counts: hl.expr.DictExpression,
    ) -> hl.expr.Float32Expression:
        n = gt_counts.get(0, 0) + gt_counts.get(1, 0) + gt_counts.get(2, 0)
        p = (2 * gt_counts.get(0, 0) + gt_counts.get(1, 0)) / (2 * n)
        q = (2 * gt_counts.get(2, 0) + gt_counts.get(1, 0)) / (2 * n)
        return 1 - (gt_counts.get(1, 0) / (2 * p * q * n))

    if callstats_expr is not None:
        # Check that AC, AN, and homozygote count are all ints
        if not (
            (
                (callstats_expr.AC.dtype == hl.tint32)
                | (callstats_expr.AC.dtype == hl.tint64)
            )
            & (
                (callstats_expr.AN.dtype == hl.tint32)
                | (callstats_expr.AN.dtype == hl.tint64)
            )
            & (
                (callstats_expr.homozygote_count.dtype == hl.tint32)
                | (callstats_expr.homozygote_count.dtype == hl.tint64)
            )
        ):
            raise ValueError(
                "callstats_expr must be a StructExpression containing fields 'AC',"
                " 'AN', and 'homozygote_count' of types int32 or int64."
            )
        n = callstats_expr.AN / 2
        q = callstats_expr.AC / callstats_expr.AN
        p = 1 - q
        return 1 - (callstats_expr.AC - (2 * callstats_expr.homozygote_count)) / (
            2 * p * q * n
        )
    else:
        return hl.bind(inbreeding_coeff, hl.agg.counter(call.n_alt_alleles()))


def fs_from_sb(
    sb: Union[hl.expr.ArrayNumericExpression, hl.expr.ArrayExpression],
    normalize: bool = True,
    min_cell_count: int = 200,
    min_count: int = 4,
    min_p_value: float = 1e-320,
) -> hl.expr.Int64Expression:
    """
    Compute `FS` (Fisher strand balance) annotation from  the `SB` (strand balance table) field.

    `FS` is the phred-scaled value of the double-sided Fisher exact test on strand balance.

    Using default values will have the same behavior as the GATK implementation, that is:
    - If sum(counts) > 2*`min_cell_count` (default to GATK value of 200), they are normalized
    - If sum(counts) < `min_count` (default to GATK value of 4), returns missing
    - Any p-value < `min_p_value` (default to GATK value of 1e-320) is truncated to that value

    In addition to the default GATK behavior, setting `normalize` to `False` will perform a chi-squared test
    for large counts (> `min_cell_count`) instead of normalizing the cell values.

    .. note::

        This function can either take
        - an array of length four containing the forward and reverse strands' counts of ref and alt alleles: [ref fwd, ref rev, alt fwd, alt rev]
        - a two dimensional array with arrays of length two, containing the counts: [[ref fwd, ref rev], [alt fwd, alt rev]]

    GATK code here: https://github.com/broadinstitute/gatk/blob/master/src/main/java/org/broadinstitute/hellbender/tools/walkers/annotator/FisherStrand.java

    :param sb: Count of ref/alt reads on each strand
    :param normalize: Whether to normalize counts is sum(counts) > min_cell_count (normalize=True), or use a chi sq instead of FET (normalize=False)
    :param min_cell_count: Maximum count for performing a FET
    :param min_count: Minimum total count to output FS (otherwise null it output)
    :return: FS value
    """
    if not isinstance(sb, hl.expr.ArrayNumericExpression):
        sb = hl.bind(lambda x: hl.flatten(x), sb)

    sb_sum = hl.bind(lambda x: hl.sum(x), sb)

    # Normalize table if counts get too large
    if normalize:
        fs_expr = hl.bind(
            lambda sb, sb_sum: hl.if_else(
                sb_sum <= 2 * min_cell_count,
                sb,
                sb.map(lambda x: hl.int(x / (sb_sum / min_cell_count))),
            ),
            sb,
            sb_sum,
        )

        # FET
        fs_expr = to_phred(
            hl.max(
                hl.fisher_exact_test(
                    fs_expr[0], fs_expr[1], fs_expr[2], fs_expr[3]
                ).p_value,
                min_p_value,
            )
        )
    else:
        fs_expr = to_phred(
            hl.max(
                hl.contingency_table_test(
                    sb[0], sb[1], sb[2], sb[3], min_cell_count=min_cell_count
                ).p_value,
                min_p_value,
            )
        )

    # Return null if counts <= `min_count`
    return hl.or_missing(
        sb_sum > min_count, hl.max(0, fs_expr)  # Needed to avoid -0.0 values
    )


def sor_from_sb(
    sb: Union[hl.expr.ArrayNumericExpression, hl.expr.ArrayExpression],
) -> hl.expr.Float64Expression:
    """
    Compute `SOR` (Symmetric Odds Ratio test) annotation from  the `SB` (strand balance table) field.

    .. note::

        This function can either take
        - an array of length four containing the forward and reverse strands' counts of ref and alt alleles: [ref fwd, ref rev, alt fwd, alt rev]
        - a two dimensional array with arrays of length two, containing the counts: [[ref fwd, ref rev], [alt fwd, alt rev]]

    GATK code here: https://github.com/broadinstitute/gatk/blob/master/src/main/java/org/broadinstitute/hellbender/tools/walkers/annotator/StrandOddsRatio.java

    :param sb: Count of ref/alt reads on each strand
    :return: SOR value
    """
    if not isinstance(sb, hl.expr.ArrayNumericExpression):
        sb = hl.bind(lambda x: hl.flatten(x), sb)

    sb = sb.map(lambda x: hl.float64(x) + 1)

    ref_fw = sb[0]
    ref_rv = sb[1]
    alt_fw = sb[2]
    alt_rv = sb[3]
    symmetrical_ratio = ((ref_fw * alt_rv) / (alt_fw * ref_rv)) + (
        (alt_fw * ref_rv) / (ref_fw * alt_rv)
    )
    ref_ratio = hl.min(ref_rv, ref_fw) / hl.max(ref_rv, ref_fw)
    alt_ratio = hl.min(alt_fw, alt_rv) / hl.max(alt_fw, alt_rv)
    sor = hl.log(symmetrical_ratio) + hl.log(ref_ratio) - hl.log(alt_ratio)

    return sor


def pab_max_expr(
    gt_expr: hl.expr.CallExpression,
    ad_expr: hl.expr.ArrayExpression,
    la_expr: Optional[hl.expr.ArrayExpression] = None,
    n_alleles_expr: Optional[hl.expr.Int32Expression] = None,
) -> hl.expr.ArrayExpression:
    """
    Compute the maximum p-value of the binomial test for the alternate allele balance (PAB) for each allele.

    .. note::

        This function can take a `gt_expr` and `ad_expr` that use local or global
        alleles. If they use local alleles, `la_expr` and `n_alleles_expr` should be
        provided to transform `gt_expr` and `ad_expr` to global alleles.

    :param gt_expr: Genotype call expression.
    :param ad_expr: Allele depth expression.
    :param la_expr: Allele local index expression. When provided `gt_expr` and
        `ad_expr` are transformed from using local alleles to global alleles using
        `la_expr`.
    :param n_alleles_expr: Number of alleles expression. Required when 'la_expr' is
        provided.
    :return: Array expression of maximum p-values.
    """
    if la_expr is not None:
        if n_alleles_expr is None:
            raise ValueError("Must provide `n_alleles_expr` if `la_expr` is provided!")

        ad_expr = hl.vds.local_to_global(
            ad_expr, la_expr, n_alleles_expr, fill_value=0, number="R"
        )
        gt_expr = hl.vds.lgt_to_gt(gt_expr, la_expr)

    expr = hl.agg.array_agg(
        lambda x: hl.agg.filter(
            gt_expr.is_het(),
            hl.agg.max(hl.binom_test(x, hl.sum(ad_expr), 0.5, "two-sided")),
        ),
        ad_expr[1:],  # Skip ref allele
    )

    return expr


def bi_allelic_expr(t: Union[hl.Table, hl.MatrixTable]) -> hl.expr.BooleanExpression:
    """
    Return a boolean expression selecting bi-allelic sites only, accounting for whether the input MT/HT was split.

    :param t: Input HT/MT
    :return: Boolean expression selecting only bi-allelic sites
    """
    return ~t.was_split if "was_split" in t.row else (hl.len(t.alleles) == 2)


def unphase_call_expr(call_expr: hl.expr.CallExpression) -> hl.expr.CallExpression:
    """
    Generate unphased version of a call expression (which can be phased or not).

    :param call_expr: Input call expression
    :return: unphased call expression
    """
    return (
        hl.case()
        .when(call_expr.is_diploid(), hl.call(call_expr[0], call_expr[1], phased=False))
        .when(call_expr.is_haploid(), hl.call(call_expr[0], phased=False))
        .default(hl.null(hl.tcall))
    )


def region_flag_expr(
    t: Union[hl.Table, hl.MatrixTable],
    non_par: bool = True,
    prob_regions: Dict[str, hl.Table] = None,
) -> hl.expr.StructExpression:
    """
    Create a `region_flag` struct that contains flags for problematic regions (i.e., LCR, decoy, segdup, and nonpar regions).

    .. note:: No hg38 resources for decoy or self chain are available yet.

    :param t: Input Table/MatrixTable
    :param non_par: If True, flag loci that occur within pseudoautosomal regions on sex chromosomes
    :param prob_regions: If supplied, flag loci that occur within regions defined in Hail Table(s)
    :return: `region_flag` struct row annotation
    """
    prob_flags_expr = (
        {"non_par": t.locus.in_x_nonpar() | t.locus.in_y_nonpar()} if non_par else {}
    )

    if prob_regions is not None:
        prob_flags_expr.update(
            {
                region_name: hl.is_defined(region_table[t.locus])
                for region_name, region_table in prob_regions.items()
            }
        )

    return hl.struct(**prob_flags_expr)


def missing_callstats_expr() -> hl.expr.StructExpression:
    """
    Create a missing callstats struct for insertion into frequency annotation arrays when data is missing.

    :return: Hail Struct with missing values for each callstats element
    """
    return hl.struct(
        AC=hl.missing(hl.tint32),
        AF=hl.missing(hl.tfloat64),
        AN=hl.missing(hl.tint32),
        homozygote_count=hl.missing(hl.tint32),
    )


def set_female_y_metrics_to_na_expr(
    t: Union[hl.Table, hl.MatrixTable],
    freq_expr: Union[hl.expr.ArrayExpression, str] = "freq",
    freq_meta_expr: Union[hl.expr.ArrayExpression, str] = "freq_meta",
    freq_index_dict_expr: Union[hl.expr.DictExpression, str] = "freq_index_dict",
) -> hl.expr.ArrayExpression:
    """
    Set Y-variant frequency callstats for female-specific metrics to missing structs.

    :param t: Table or MatrixTable for which to adjust female metrics.
    :param freq_expr: Array expression or string annotation name for the frequency
        array. Default is "freq".
    :param freq_meta_expr: Array expression or string annotation name for the frequency
        metadata. Default is "freq_meta".
    :param freq_index_dict_expr: Dict expression or string annotation name for the
        frequency metadata index dictionary. Default is "freq_index_dict".
    :return: Hail array expression to set female Y-variant metrics to missing values.
    """
    if isinstance(freq_expr, str):
        freq_expr = t[freq_expr]
    if isinstance(freq_meta_expr, str):
        freq_meta_expr = t[freq_meta_expr]
    if isinstance(freq_index_dict_expr, str):
        freq_index_dict_expr = t[freq_index_dict_expr]

    female_idx = hl.map(
        lambda x: freq_index_dict_expr[x],
        hl.filter(lambda x: x.contains("XX"), freq_index_dict_expr.keys()),
    )
    freq_idx_range = hl.range(hl.len(freq_meta_expr))

    new_freq_expr = hl.if_else(
        (t.locus.in_y_nonpar() | t.locus.in_y_par()),
        hl.map(
            lambda x: hl.if_else(
                female_idx.contains(x), missing_callstats_expr(), freq_expr[x]
            ),
            freq_idx_range,
        ),
        freq_expr,
    )

    return new_freq_expr


def hemi_expr(
    locus: hl.expr.LocusExpression,
    sex_expr: hl.expr.StringExpression,
    gt: hl.expr.CallExpression,
    male_str: str = "XY",
) -> hl.expr.BooleanExpression:
    """
    Return whether genotypes are hemizygous.

    Return missing expression if locus is not in chrX/chrY non-PAR regions.

    :param locus: Input locus.
    :param sex_expr: Input StringExpression indicating whether sample is XX or XY.
    :param gt: Input genotype.
    :param xy_str: String indicating whether sample is XY. Default is "XY".
    :return: BooleanExpression indicating whether genotypes are hemizygous.
    """
    return hl.or_missing(
        locus.in_x_nonpar() | locus.in_y_nonpar(),
        # Haploid genotypes have a single integer, so checking if
        # mt.GT[0] is alternate allele
        gt.is_haploid() & (sex_expr == male_str) & (gt[0] == 1),
    )


def merge_freq_arrays(
    farrays: List[hl.expr.ArrayExpression],
    fmeta: List[List[Dict[str, str]]],
    operation: str = "sum",
    set_negatives_to_zero: bool = False,
    count_arrays: Optional[Dict[str, List[hl.expr.ArrayExpression]]] = None,
) -> Union[
    Tuple[hl.expr.ArrayExpression, List[Dict[str, int]]],
    Tuple[
        hl.expr.ArrayExpression,
        List[Dict[str, int]],
        Dict[str, List[hl.expr.ArrayExpression]],
    ],
]:
    """
    Merge a list of frequency arrays based on the supplied `operation`.

    .. warning::
        Arrays must be on the same Table.

    .. note::

        Arrays do not have to contain the same groupings or order of groupings but
        the array indices for a freq array in `farrays` must be the same as its associated
        frequency metadata index in `fmeta` i.e., `farrays = [freq1, freq2]` then `fmeta`
        must equal `[fmeta1, fmeta2]` where fmeta1 contains the metadata information
        for freq1.

        If `operation` is set to "sum", groups in the merged array
        will be the union of groupings found within the arrays' metadata and all arrays
        with be summed by grouping. If `operation` is set to "diff", the merged array
        will contain groups only found in the first array of `fmeta`. Any array containing
        any of these groups will have thier values subtracted from the values of the first array.

    :param farrays: List of frequency arrays to merge. First entry in the list is the primary array to which other arrays will be added or subtracted. All arrays must be on the same Table.
    :param fmeta: List of frequency metadata for arrays being merged.
    :param operation: Merge operation to perform. Options are "sum" and "diff". If "diff" is passed, the first freq array in the list will have the other arrays subtracted from it.
    :param set_negatives_to_zero: If True, set negative array values to 0 for AC, AN, AF, and homozygote_count. If False, raise a ValueError. Default is False.
    :param count_arrays: Dictionary of Lists of arrays containing counts to merge using the passed operation. Must use the same group indexing as fmeta. Keys are the descriptor names, values are Lists of arrays to merge. Default is None.
    :return: Tuple of merged frequency array, frequency metadata list and if `count_arrays` is not None, a dictionary of merged count arrays.
    """
    if len(farrays) < 2:
        raise ValueError("Must provide at least two frequency arrays to merge!")
    if len(farrays) != len(fmeta):
        raise ValueError("Length of farrays and fmeta must be equal!")
    if operation not in ["sum", "diff"]:
        raise ValueError("Operation must be either 'sum' or 'diff'!")
    if count_arrays is not None:
        for k, count_array in count_arrays.items():
            if len(count_array) != len(fmeta):
                raise ValueError(
                    f"Length of  count_array '{k}' and fmeta must be equal!"
                )

    # Create a list where each entry is a dictionary whose key is an aggregation
    # group and the value is the corresponding index in the freq array.
    fmeta = [hl.dict(hl.enumerate(f).map(lambda x: (x[1], [x[0]]))) for f in fmeta]
    all_keys = hl.fold(lambda i, j: (i | j.key_set()), fmeta[0].key_set(), fmeta[1:])

    # Merge dictionaries in the list into a single dictionary where key is aggregation
    # group and the value is a list of the group's index in each of the freq arrays, if
    # it exists. For "sum" operation, use keys, aka groups, found in all freq dictionaries.
    # For "diff" operations, only use key_set from the first entry.
    fmeta = hl.fold(
        lambda i, j: hl.dict(
            (hl.if_else(operation == "sum", all_keys, i.key_set())).map(
                lambda k: (
                    k,
                    i.get(k, [hl.missing(hl.tint32)]).extend(
                        j.get(k, [hl.missing(hl.tint32)])
                    ),
                )
            )
        ),
        fmeta[0],
        fmeta[1:],
    )

    # Create a list of tuples from the dictionary, sorted by the list of indices for
    # each aggregation group.
    fmeta = hl.sorted(fmeta.items(), key=lambda f: f[1])

    # Create a list of the aggregation groups, maintaining the sorted order.
    new_freq_meta = fmeta.map(lambda x: x[0])

    # Create array for each aggregation group of arrays containing the group's freq
    # values from each freq array.
    freq_meta_idx = fmeta.map(lambda x: hl.zip(farrays, x[1]).map(lambda i: i[0][i[1]]))

    def _sum_or_diff_fields(
        field_1_expr: str, field_2_expr: str
    ) -> hl.expr.Int32Expression:
        """
        Sum or subtract fields in call statistics struct.

        :param field_1_expr: First field to sum or diff.
        :param field_2_expr: Second field to sum or diff.
        :return: Merged field value.
        """
        return hl.if_else(
            operation == "sum",
            hl.or_else(field_1_expr, 0) + hl.or_else(field_2_expr, 0),
            hl.or_else(field_1_expr, 0) - hl.or_else(field_2_expr, 0),
        )

    # Iterate through the groups and their freq lists to merge callstats.
    callstat_ann = ["AC", "AN", "homozygote_count"]
    callstat_ann_af = ["AC", "AF", "AN", "homozygote_count"]
    new_freq = freq_meta_idx.map(
        lambda x: hl.bind(
            lambda y: y.annotate(AF=hl.or_missing(y.AN > 0, y.AC / y.AN)).select(
                *callstat_ann_af
            ),
            hl.fold(
                lambda i, j: hl.struct(
                    **{ann: _sum_or_diff_fields(i[ann], j[ann]) for ann in callstat_ann}
                ),
                x[0].select(*callstat_ann),
                x[1:],
            ),
        )
    )
    # Create count_array_meta_idx using the fmeta then iterate through each group
    # in the list of tuples to access each group's entry per array. Sum or diff the
    # values for each group across arrays to make a new_counts_array annotation.
    if count_arrays:
        new_counts_array_dict = {}
        for k, count_array in count_arrays.items():
            count_array_meta_idx = fmeta.map(
                lambda x: hl.zip(count_array, x[1]).map(lambda i: i[0][i[1]])
            )

            new_counts_array_dict[k] = count_array_meta_idx.map(
                lambda x: hl.fold(
                    lambda i, j: _sum_or_diff_fields(i, j),
                    x[0],
                    x[1:],
                ),
            )
    # Check and see if any annotation within the merged array is negative. If so,
    # raise an error if set_negatives_to_zero is False or set the value to 0 if
    # set_negatives_to_zero is True.
    if operation == "diff":
        negative_value_error_msg = (
            "Negative values found in merged %s array. Review data or set"
            " `set_negatives_to_zero` to True to set negative values to 0."
        )
        callstat_ann.append("AF")
        new_freq = new_freq.map(
            lambda x: x.annotate(
                **{
                    ann: (
                        hl.case()
                        .when(set_negatives_to_zero, hl.max(x[ann], 0))
                        .when(x[ann] >= 0, x[ann])
                        .or_error(negative_value_error_msg % "freq")
                    )
                    for ann in callstat_ann
                }
            )
        )
        if count_arrays:
            for k, new_counts_array in new_counts_array_dict.items():
                new_counts_array_dict[k] = new_counts_array.map(
                    lambda x: hl.case()
                    .when(set_negatives_to_zero, hl.max(x, 0))
                    .when(x >= 0, x)
                    .or_error(negative_value_error_msg % "counts")
                )

    new_freq_meta = hl.eval(new_freq_meta)
    if count_arrays:
        return new_freq, new_freq_meta, new_counts_array_dict
    else:
        return new_freq, new_freq_meta


def merge_histograms(hists: List[hl.expr.StructExpression]) -> hl.expr.Expression:
    """
    Merge a list of histogram annotations.

    This function merges a list of histogram annotations by summing the arrays
    in an element-wise fashion. It keeps one 'bin_edge' annotation but merges the
    'bin_freq', 'n_smaller', and 'n_larger' annotations by summing them.

    .. note::

        Bin edges are assumed to be the same for all histograms.

    :param hists: List of histogram structs to merge.
    :return: Merged histogram struct.
    """
    return hl.fold(
        lambda i, j: hl.struct(
            **{
                "bin_edges": hl.or_else(i.bin_edges, j.bin_edges),
                "bin_freq": hl.zip(
                    hl.or_else(i.bin_freq, hl.literal([hl.missing(hl.tint)])),
                    hl.or_else(j.bin_freq, hl.literal([hl.missing(hl.tint)])),
                    fill_missing=True,
                ).map(lambda x: hl.or_else(x[0], 0) + hl.or_else(x[1], 0)),
                "n_smaller": hl.or_else(i.n_smaller, 0) + hl.or_else(j.n_smaller, 0),
                "n_larger": hl.or_else(i.n_larger, 0) + hl.or_else(j.n_larger, 0),
            }
        ),
        hists[0].select("bin_edges", "bin_freq", "n_smaller", "n_larger"),
        hists[1:],
    )


# Functions used for computing allele frequency.
def annotate_freq(
    mt: hl.MatrixTable,
    sex_expr: Optional[hl.expr.StringExpression] = None,
    pop_expr: Optional[hl.expr.StringExpression] = None,
    subpop_expr: Optional[hl.expr.StringExpression] = None,
    additional_strata_expr: Optional[
        Union[
            List[Dict[str, hl.expr.StringExpression]],
            Dict[str, hl.expr.StringExpression],
        ]
    ] = None,
    downsamplings: Optional[List[int]] = None,
    downsampling_expr: Optional[hl.expr.StructExpression] = None,
    ds_pop_counts: Optional[Dict[str, int]] = None,
    entry_agg_funcs: Optional[Dict[str, Tuple[Callable, Callable]]] = None,
    annotate_mt: bool = True,
) -> Union[hl.Table, hl.MatrixTable]:
    """
    Annotate `mt` with stratified allele frequencies.

    The output Matrix table will include:
        - row annotation `freq` containing the stratified allele frequencies
        - global annotation `freq_meta` with metadata
        - global annotation `freq_meta_sample_count` with sample count information

    .. note::

        Currently this only supports bi-allelic sites.

        The input `mt` needs to have the following entry fields:
          - GT: a CallExpression containing the genotype
          - adj: a BooleanExpression containing whether the genotype is of high quality
            or not.

        All expressions arguments need to be expression on the input `mt`.

    .. rubric:: `freq` row annotation

    The `freq` row annotation is an Array of Structs, with each Struct containing the
    following fields:

        - AC: int32
        - AF: float64
        - AN: int32
        - homozygote_count: int32

    Each element of the array corresponds to a stratification of the data, and the
    metadata about these annotations is stored in the globals.

    .. rubric:: Global `freq_meta` metadata annotation

    The global annotation `freq_meta` is added to the input `mt`. It is a list of dict.
    Each element of the list contains metadata on a frequency stratification and the
    index in the list corresponds to the index of that frequency stratification in the
    `freq` row annotation.

    .. rubric:: Global `freq_meta_sample_count` annotation

    The global annotation `freq_meta_sample_count` is added to the input `mt`. This is a
    sample count per sample grouping defined in the `freq_meta` global annotation.

    .. rubric:: The `additional_strata_expr` parameter

    If the `additional_strata_expr` parameter is used, frequencies will be computed for
    each of the strata dictionaries across all values. For example, if
    `additional_strata_expr` is set to `[{'platform': mt.platform},
    {'platform':mt.platform, 'pop': mt.pop}, {'age_bin': mt.age_bin}]`, then
    frequencies will be computed for each of the values of `mt.platform`, each of the
    combined values of `mt.platform` and `mt.pop`, and each of the values of
    `mt.age_bin`.

    .. rubric:: The `downsamplings` parameter

    If the `downsamplings` parameter is used without the `downsampling_expr`,
    frequencies will be computed for all samples and by population (if `pop_expr` is
    specified) by downsampling the number of samples without replacement to each of the
    numbers specified in the `downsamplings` array, provided that there are enough
    samples in the dataset. In addition, if `pop_expr` is specified, a downsampling to
    each of the exact number of samples present in each population is added. Note that
    samples are randomly sampled only once, meaning that the lower downsamplings are
    subsets of the higher ones. If the `downsampling_expr` parameter is used with the
    `downsamplings` parameter, the `downsamplings` parameter informs the function which
    downsampling groups were already created and are to be used in the frequency
    calculation.

    .. rubric:: The `downsampling_expr` and `ds_pop_counts` parameters

    If the `downsampling_expr` parameter is used, `downsamplings` must also be set
    and frequencies will be computed for all samples and by population (if `pop_expr`
    is specified) using the downsampling indices to each of the numbers specified in
    the `downsamplings` array. The function expects a 'global_idx', and if `pop_expr`
    is used, a 'pop_idx' within the `downsampling_expr` to be used to determine if a
    sample belongs within a certain downsampling group, i.e. the index is less than
    the group size. `The function `annotate_downsamplings` can be used to to create
    the `downsampling_expr`, `downsamplings`, and `ds_pop_counts` expressions.

    .. rubric:: The `entry_agg_funcs` parameter

    If the `entry_agg_funcs` parameter is used, the output MatrixTable will also
    contain the annotations specified in the `entry_agg_funcs` parameter. The keys of
    the dict are the names of the annotations and the values are tuples of functions.
    The first function is used to transform the `mt` entries in some way, and the
    second function is used to aggregate the output from the first function. For
    example, if `entry_agg_funcs` is set to {'adj_samples': (get_adj_expr, hl.agg.sum)}`,
    then the output MatrixTable will contain an annotation `adj_samples` which is an
    array of the number of adj samples per strata in each row.

    :param mt: Input MatrixTable
    :param sex_expr: When specified, frequencies are stratified by sex. If `pop_expr`
        is also specified, then a pop/sex stratifiction is added.
    :param pop_expr: When specified, frequencies are stratified by population. If
        `sex_expr` is also specified, then a pop/sex stratifiction is added.
    :param subpop_expr: When specified, frequencies are stratified by sub-continental
        population. Note that `pop_expr` is required as well when using this option.
    :param additional_strata_expr: When specified, frequencies are stratified by the
        given additional strata. This can e.g. be used to stratify by platform,
        platform-pop, platform-pop-sex.
    :param downsamplings: When specified, frequencies are computed by downsampling the
        data to the number of samples given in the list. Note that if `pop_expr` is
        specified, downsamplings by population is also computed.
    :param downsampling_expr: When specified, frequencies are computed using the
        downsampling indices in the provided StructExpression. Note that if `pop_idx`
        is specified within the struct, downsamplings by population is also computed.
    :param ds_pop_counts: When specified, frequencies are computed by downsampling the
        data to the number of samples per pop in the dict. The key is the population
        and the value is the number of samples.
    :param entry_agg_funcs: When specified, additional annotations are added to the
        output Table/MatrixTable. The keys of the dict are the names of the annotations
        and the values are tuples of functions. The first function is used to transform
        the `mt` entries in some way, and the second function is used to aggregate the
        output from the first function.
    :param annotate_mt: Whether to return the full MatrixTable with annotations added
        instead of only a Table with `freq` and other annotations. Default is True.
    :return: MatrixTable or Table with `freq` annotation.
    """
    errors = []
    if downsampling_expr is not None:
        if downsamplings is None:
            errors.append(
                "annotate_freq requires `downsamplings` when using `downsampling_expr`"
            )
        if downsampling_expr.get("pop_idx") is not None:
            if ds_pop_counts is None:
                errors.append(
                    "annotate_freq requires `ds_pop_counts` when using "
                    "`downsampling_expr` with pop_idx"
                )
    if errors:
        raise ValueError("The following errors were found: \n" + "\n".join(errors))

    # Generate downsamplings and assign downsampling_expr if it is None when
    # downsamplings is supplied.
    if downsamplings is not None and downsampling_expr is None:
        ds_ht = annotate_downsamplings(mt, downsamplings, pop_expr=pop_expr).cols()
        downsamplings = hl.eval(ds_ht.downsamplings)
        ds_pop_counts = hl.eval(ds_ht.ds_pop_counts)
        downsampling_expr = ds_ht[mt.col_key].downsampling

    # Build list of all stratification groups to be used in the frequency calculation.
    strata_expr = build_freq_stratification_list(
        sex_expr=sex_expr,
        pop_expr=pop_expr,
        subpop_expr=subpop_expr,
        additional_strata_expr=additional_strata_expr,
        downsampling_expr=downsampling_expr,
    )

    # Annotate the MT cols with each of the expressions in strata_expr and redefine
    # strata_expr based on the column HT with added annotations.
    ht = mt.annotate_cols(**{k: v for d in strata_expr for k, v in d.items()}).cols()
    strata_expr = [{k: ht[k] for k in d} for d in strata_expr]

    # Annotate HT with a freq_meta global and group membership array for each sample
    # indicating whether the sample belongs to the group defined by freq_meta elements.
    ht = generate_freq_group_membership_array(
        ht,
        strata_expr,
        downsamplings=downsamplings,
        ds_pop_counts=ds_pop_counts,
    )

    freq_ht = compute_freq_by_strata(
        mt.annotate_cols(group_membership=ht[mt.col_key].group_membership),
        entry_agg_funcs=entry_agg_funcs,
    )
    freq_ht = freq_ht.annotate_globals(**ht.index_globals())

    if annotate_mt:
        mt = mt.annotate_rows(**freq_ht[mt.row_key])
        mt = mt.annotate_globals(**freq_ht.index_globals())
        return mt

    else:
        return freq_ht


def annotate_downsamplings(
    t: Union[hl.MatrixTable, hl.Table],
    downsamplings: List[int],
    pop_expr: Optional[hl.expr.StringExpression] = None,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Annotate MatrixTable or Table with downsampling groups.

    :param t: Input MatrixTable or Table.
    :param downsamplings: List of downsampling sizes.
    :param pop_expr: Optional expression for population group. When provided, population
        sample sizes are added as values to downsamplings.
    :return: MatrixTable or Table with downsampling annotations.
    """
    if isinstance(t, hl.MatrixTable):
        if pop_expr is not None:
            ht = t.annotate_cols(pop=pop_expr).cols()
        else:
            ht = t.cols()
    else:
        if pop_expr is not None:
            ht = t.annotate(pop=pop_expr)
        else:
            ht = t

    ht = ht.key_by(r=hl.rand_unif(0, 1))

    # Add a global index for use in computing frequencies, or other aggregate stats on
    # the downsamplings.
    scan_expr = {"global_idx": hl.scan.count()}

    # If pop_expr is provided, add all pop counts to the downsamplings list.
    if pop_expr is not None:
        pop_counts = ht.aggregate(
            hl.agg.filter(hl.is_defined(ht.pop), hl.agg.counter(ht.pop))
        )
        downsamplings = [x for x in downsamplings if x <= sum(pop_counts.values())]
        downsamplings = sorted(set(downsamplings + list(pop_counts.values())))
        # Add an index by pop for use in computing frequencies, or other aggregate stats
        # on the downsamplings.
        scan_expr["pop_idx"] = hl.scan.counter(ht.pop).get(ht.pop, 0)
    else:
        pop_counts = None
    logger.info("Found %i downsamplings: %s", len(downsamplings), downsamplings)

    ht = ht.annotate(**scan_expr)
    ht = ht.key_by("s").select(*scan_expr)

    if isinstance(t, hl.MatrixTable):
        t = t.annotate_cols(downsampling=ht[t.s])
    else:
        t = t.annotate(downsampling=ht[t.s])

    t = t.annotate_globals(
        downsamplings=downsamplings,
        ds_pop_counts=pop_counts,
    )

    return t


def build_freq_stratification_list(
    sex_expr: Optional[hl.expr.StringExpression] = None,
    pop_expr: Optional[hl.expr.StringExpression] = None,
    subpop_expr: Optional[hl.expr.StringExpression] = None,
    additional_strata_expr: Optional[
        Union[
            List[Dict[str, hl.expr.StringExpression]],
            Dict[str, hl.expr.StringExpression],
        ]
    ] = None,
    downsampling_expr: Optional[hl.expr.StructExpression] = None,
) -> List[Dict[str, hl.expr.StringExpression]]:
    """
    Build a list of stratification groupings to be used in frequency calculations based on supplied parameters.

    .. note::
        This function is primarily used through `annotate_freq` but can be used
        independently if desired. The returned list of stratifications can be passed to
        `generate_freq_group_membership_array`.

    :param sex_expr: When specified, the returned list contains a stratification for
        sex. If `pop_expr` is also specified, then the returned list also contains a
        pop/sex stratification.
    :param pop_expr: When specified, the returned list contains a stratification for
        population. If `sex_expr` is also specified, then the returned list also
        contains a pop/sex stratification.
    :param subpop_expr: When specified, the returned list contains a stratification for
        sub-continental population. Note that `pop_expr` is required as well when using
        this option.
    :param additional_strata_expr: When specified, the returned list contains a
        stratification for each of the additional strata. This can e.g. be used to
        stratify by platform, platform-pop, platform-pop-sex.
    :param downsampling_expr: When specified, the returned list contains a
        stratification for downsampling. If `pop_expr` is also specified, then the
        returned list also contains a downsampling/pop stratification.
    :return: List of dictionaries specifying stratification groups where the keys of
        each dictionary are strings and the values are corresponding expressions that
        define the values to stratify frequency calculations by.
    """
    errors = []
    if subpop_expr is not None and pop_expr is None:
        errors.append("annotate_freq requires pop_expr when using subpop_expr")

    if downsampling_expr is not None:
        if downsampling_expr.get("global_idx") is None:
            errors.append(
                "annotate_freq requires `downsampling_expr` with key 'global_idx'"
            )
        if downsampling_expr.get("pop_idx") is None:
            if pop_expr is not None:
                errors.append(
                    "annotate_freq requires `downsampling_expr` with key 'pop_idx' when"
                    " using `pop_expr`"
                )
        else:
            if pop_expr is None:
                errors.append(
                    "annotate_freq requires `pop_expr` when using `downsampling_expr` "
                    "with pop_idx"
                )

    if errors:
        raise ValueError("The following errors were found: \n" + "\n".join(errors))

    # Build list of strata expressions based on supplied parameters.
    strata_expr = []
    if pop_expr is not None:
        strata_expr.append({"pop": pop_expr})
    if sex_expr is not None:
        strata_expr.append({"sex": sex_expr})
        if pop_expr is not None:
            strata_expr.append({"pop": pop_expr, "sex": sex_expr})
    if subpop_expr is not None:
        strata_expr.append({"pop": pop_expr, "subpop": subpop_expr})

    # Add downsampling to strata expressions, include pop in the strata if supplied.
    if downsampling_expr is not None:
        downsampling_strata = {"downsampling": downsampling_expr}
        if pop_expr is not None:
            downsampling_strata["pop"] = pop_expr
        strata_expr.append(downsampling_strata)

    # Add additional strata expressions.
    if additional_strata_expr is not None:
        if isinstance(additional_strata_expr, dict):
            additional_strata_expr = [additional_strata_expr]
        strata_expr.extend(additional_strata_expr)

    return strata_expr


def generate_freq_group_membership_array(
    ht: hl.Table,
    strata_expr: List[Dict[str, hl.expr.StringExpression]],
    downsamplings: Optional[List[int]] = None,
    ds_pop_counts: Optional[Dict[str, int]] = None,
    remove_zero_sample_groups: bool = False,
    no_raw_group: bool = False,
) -> hl.Table:
    """
    Generate a Table with a 'group_membership' array for each sample indicating whether the sample belongs to specific stratification groups.

    .. note::
        This function is primarily used through `annotate_freq` but can be used
        independently if desired. Please see the `annotate_freq` function for more
        complete documentation.

    The following global annotations are added to the returned Table:
        - freq_meta: Each element of the list contains metadata on a stratification
          group.
        - freq_meta_sample_count: sample count per grouping defined in `freq_meta`.
        - If downsamplings or ds_pop_counts are specified, they are also added as
          global annotations on the returned Table.

    Each sample is annotated with a 'group_membership' array indicating whether the
    sample belongs to specific stratification groups. All possible value combinations
    are determined for each stratification grouping in the `strata_expr` list.

    :param ht: Input Table that contains Expressions specified by `strata_expr`.
    :param strata_expr: List of dictionaries specifying stratification groups where
        the keys of each dictionary are strings and the values are corresponding
        expressions that define the values to stratify frequency calculations by.
    :param downsamplings: List of downsampling values to include in the stratifications.
    :param ds_pop_counts: Dictionary of population counts for each downsampling value.
    :param remove_zero_sample_groups: Whether to remove groups with a sample count of 0.
        Default is False.
    :param no_raw_group: Whether to remove the raw group from the 'group_membership'
        annotation and the 'freq_meta' and 'freq_meta_sample_count' global annotations.
        Default is False.
    :return: Table with the 'group_membership' array annotation.
    """
    errors = []
    ds_in_strata = any("downsampling" in s for s in strata_expr)
    global_idx_in_ds_expr = any(
        "global_idx" in s["downsampling"] for s in strata_expr if "downsampling" in s
    )
    pop_in_strata = any("pop" in s for s in strata_expr)
    pop_idx_in_ds_expr = any(
        "pop_idx" in s["downsampling"]
        for s in strata_expr
        if "downsampling" in s and ds_pop_counts is not None
    )

    if downsamplings is not None and not ds_in_strata:
        errors.append(
            "Strata must contain a downsampling expression when downsamplings"
            "are provided."
        )
    if downsamplings is not None and not global_idx_in_ds_expr:
        errors.append(
            "Strata must contain a downsampling expression with 'global_idx' when "
            "downsamplings are provided."
        )
    if ds_pop_counts is not None and not pop_in_strata:
        errors.append(
            "Strata must contain a population expression 'pop' when ds_pop_counts "
            " are provided."
        )
    if ds_pop_counts is not None and not pop_idx_in_ds_expr:
        errors.append(
            "Strata must contain a downsampling expression with 'pop_idx' when "
            "ds_pop_counts are provided."
        )

    if errors:
        raise ValueError("The following errors were found: \n" + "\n".join(errors))

    # Get counters for all strata.
    strata_counts = ht.aggregate(
        hl.struct(
            **{
                k: hl.agg.filter(hl.is_defined(v), hl.agg.counter({k: v}))
                for strata in strata_expr
                for k, v in strata.items()
            }
        )
    )

    # Add all desired strata to sample group filters.
    sample_group_filters = [({}, True)]
    for strata in strata_expr:
        downsampling_expr = strata.get("downsampling")
        strata_values = []
        # Add to all downsampling groups, both global and population-specific, to
        # strata.
        for s in strata:
            if s == "downsampling":
                v = [("downsampling", d) for d in downsamplings]
            else:
                v = [(s, k[s]) for k in strata_counts.get(s, {})]
                if s == "pop" and downsampling_expr is not None:
                    v.append(("pop", "global"))
            strata_values.append(v)

        # Get all combinations of strata values.
        strata_combinations = itertools.product(*strata_values)
        # Create sample group filters that are evaluated on each sample for each strata
        # combination. Strata combinations are evaluated as a logical AND, e.g.
        # {"pop":nfe, "downsampling":1000} or "nfe-10000" creates the filter expression
        # pop == nfe AND downsampling pop_idx < 10000.
        for combo in strata_combinations:
            combo = dict(combo)
            ds = combo.get("downsampling")
            pop = combo.get("pop")
            # If combo contains downsampling, determine the downsampling index
            # annotation to use.
            downsampling_idx = "global_idx"
            if ds is not None:
                if pop is not None and pop != "global":
                    # Don't include population downsamplings where the downsampling is
                    # larger than the number of samples in the population.
                    if ds > ds_pop_counts[pop]:
                        continue
                    downsampling_idx = "pop_idx"

            # If combo contains downsampling, add downsampling filter expression.
            combo_filter_exprs = []
            for s, v in combo.items():
                if s == "downsampling":
                    combo_filter_exprs.append(downsampling_expr[downsampling_idx] < v)
                else:
                    if s != "pop" or v != "global":
                        combo_filter_exprs.append(strata[s] == v)
            combo = {k: str(v) for k, v in combo.items()}
            sample_group_filters.append((combo, hl.all(combo_filter_exprs)))

    n_groups = len(sample_group_filters)
    logger.info("number of filters: %i", n_groups)

    # Get sample count per strata group.
    freq_meta_sample_count = ht.aggregate(
        [hl.agg.count_where(x[1]) for x in sample_group_filters]
    )

    if remove_zero_sample_groups:
        filter_freq = hl.enumerate(freq_meta_sample_count).filter(lambda x: x[1] > 0)
        freq_meta_sample_count = filter_freq.map(lambda x: x[1])
        idx_keep = hl.eval(filter_freq.map(lambda x: x[0]))
        sample_group_filters = [sample_group_filters[i] for i in idx_keep]

    # Annotate columns with group_membership.
    ht = ht.select(group_membership=[x[1] for x in sample_group_filters])

    # Create and annotate global expression with meta and sample count information.
    freq_meta = [
        dict(**sample_group[0], group="adj") for sample_group in sample_group_filters
    ]

    if not no_raw_group:
        # Sample group membership for the "raw" group, representing all samples, is
        # the same as the first group in the group_membership array.
        ht = ht.annotate(
            group_membership=hl.array([ht.group_membership[0]]).extend(
                ht.group_membership
            )
        )
        # Add the "raw" group, representing all samples, to the freq_meta_expr list.
        freq_meta.insert(1, {"group": "raw"})
        freq_meta_sample_count = hl.array([freq_meta_sample_count[0]]).extend(
            freq_meta_sample_count
        )

    global_expr = {
        "freq_meta": freq_meta,
        "freq_meta_sample_count": freq_meta_sample_count,
    }

    if downsamplings is not None:
        global_expr["downsamplings"] = downsamplings
    if ds_pop_counts is not None:
        global_expr["ds_pop_counts"] = ds_pop_counts

    ht = ht.select_globals(**global_expr)
    ht = ht.checkpoint(hl.utils.new_temp_file("group_membership", "ht"))

    return ht


def compute_freq_by_strata(
    mt: hl.MatrixTable,
    entry_agg_funcs: Optional[Dict[str, Tuple[Callable, Callable]]] = None,
    select_fields: Optional[List[str]] = None,
    group_membership_includes_raw_group: bool = True,
) -> hl.Table:
    """
    Compute call statistics and, when passed, entry aggregation function(s) by strata.

    The computed call statistics are AC, AF, AN, and homozygote_count. The entry
    aggregation functions are applied to the MatrixTable entries and aggregated. The
    MatrixTable must contain a 'group_membership' annotation (like the one added by
    `generate_freq_group_membership_array`) that is a list of bools to aggregate the
    columns by.

    .. note::
        This function is primarily used through `annotate_freq` but can be used
        independently if desired. Please see the `annotate_freq` function for more
        complete documentation.

    :param mt: Input MatrixTable.
    :param entry_agg_funcs: Optional dict of entry aggregation functions. When
        specified, additional annotations are added to the output Table/MatrixTable.
        The keys of the dict are the names of the annotations and the values are tuples
        of functions. The first function is used to transform the `mt` entries in some
        way, and the second function is used to aggregate the output from the first
        function.
    :param select_fields: Optional list of row fields from `mt` to keep on the output
        Table.
    :param group_membership_includes_raw_group: Whether the 'group_membership'
        annotation includes an entry for the 'raw' group, representing all samples. If
        False, the 'raw' group is inserted as the second element in all added
        annotations using the same 'group_membership', resulting
        in array lengths of 'group_membership'+1. If True, the second element of each
        added annotation is still the 'raw' group, but the group membership is
        determined by the values in the second element of 'group_membership', and the
        output annotations will be the same length as 'group_membership'. Default is
        True.
    :return: Table or MatrixTable with allele frequencies by strata.
    """
    if not group_membership_includes_raw_group:
        # Add the 'raw' group to the 'group_membership' annotation.
        mt = mt.annotate_cols(
            group_membership=hl.array([mt.group_membership[0]]).extend(
                mt.group_membership
            )
        )

    # Add adj_groups global annotation indicating that the second element in
    # group_membership is 'raw' and all others are 'adj'.
    mt = mt.annotate_globals(
        adj_groups=hl.range(hl.len(mt.group_membership.take(1)[0])).map(
            lambda x: x != 1
        )
    )

    if entry_agg_funcs is None:
        entry_agg_funcs = {}

    def _get_freq_expr(gt_expr: hl.expr.CallExpression) -> hl.expr.StructExpression:
        """
        Get struct expression with call statistics.

        :param gt_expr: CallExpression to compute call statistics on.
        :return: StructExpression with call statistics.
        """
        # Get the source Table for the CallExpression to grab alleles.
        ht = gt_expr._indices.source
        freq_expr = hl.agg.call_stats(gt_expr, ht.alleles)
        # Select non-ref allele (assumes bi-allelic).
        freq_expr = freq_expr.annotate(
            AC=freq_expr.AC[1],
            AF=freq_expr.AF[1],
            homozygote_count=freq_expr.homozygote_count[1],
        )

        return freq_expr

    entry_agg_funcs["freq"] = (lambda x: x.GT, _get_freq_expr)

    return agg_by_strata(mt, entry_agg_funcs, select_fields).drop("adj_groups")


def agg_by_strata(
    mt: hl.MatrixTable,
    entry_agg_funcs: Dict[str, Tuple[Callable, Callable]],
    select_fields: Optional[List[str]] = None,
    group_membership_ht: Optional[hl.Table] = None,
    entry_agg_group_membership: Optional[Dict[str, List[dict]]] = None,
) -> hl.Table:
    """
    Get row expression for annotations of each entry aggregation function(s) by strata.

    The entry aggregation functions are applied to the MatrixTable entries and
    aggregated. If no `group_membership_ht` (like the one returned by
    `generate_freq_group_membership_array`) is supplied, `mt` must contain a
    'group_membership' annotation that is a list of bools to aggregate the columns by.

    :param mt: Input MatrixTable.
    :param entry_agg_funcs: Dict of entry aggregation functions where the
        keys of the dict are the names of the annotations and the values are tuples
        of functions. The first function is used to transform the `mt` entries in some
        way, and the second function is used to aggregate the output from the first
        function.
    :param select_fields: Optional list of row fields from `mt` to keep on the output
        Table.
    :param group_membership_ht: Optional Table containing group membership annotations
        to stratify the aggregations by. If not provided, the 'group_membership'
        annotation is expected to be present on `mt`.
    :param entry_agg_group_membership: Optional dict indicating the subset of group
        strata in 'freq_meta' to run the entry aggregation functions on. The keys of
        the dict can be any of the keys in `entry_agg_funcs` and the values are lists
        of dicts. Each dict in the list contains the strata in 'freq_meta' to use for
        the corresponding entry aggregation function. If provided, 'freq_meta' must be
        present in `group_membership_ht` or `mt` and represent the same strata as those
        in 'group_membership'. If not provided, all entries of the 'group_membership'
        annotation will have the entry aggregation functions applied to them.
    :return: Table with annotations of stratified aggregations.
    """
    if group_membership_ht is None and "group_membership" not in mt.col:
        raise ValueError(
            "The 'group_membership' annotation is not found in the input MatrixTable "
            "and 'group_membership_ht' is not specified."
        )

    if select_fields is None:
        select_fields = []

    if group_membership_ht is None:
        logger.info(
            "'group_membership_ht' is not specified, using sample stratification "
            "indicated by the 'group_membership' annotation on the input MatrixTable."
        )
        group_globals = mt.index_globals()
    else:
        logger.info(
            "'group_membership_ht' is specified, using sample stratification indicated "
            "by its 'group_membership' annotation."
        )
        group_globals = group_membership_ht.index_globals()
        mt = mt.annotate_cols(
            group_membership=group_membership_ht[mt.col_key].group_membership
        )

    global_expr = {}
    n_groups = len(mt.group_membership.take(1)[0])
    if "adj_groups" in group_globals:
        logger.info(
            "Using the 'adj_groups' global annotation to determine adj filtered "
            "stratification groups."
        )
        global_expr["adj_groups"] = group_globals.adj_groups
    elif "freq_meta" in group_globals:
        logger.info(
            "No 'adj_groups' global annotation found, using the 'freq_meta' global "
            "annotation to determine adj filtered stratification groups."
        )
        global_expr["adj_groups"] = group_globals.freq_meta.map(
            lambda x: x.get("group", "NA") == "adj"
        )
    else:
        logger.info(
            "No 'adj_groups' or 'freq_meta' global annotations found. All groups will "
            "be considered non-adj."
        )
        global_expr["adj_groups"] = hl.range(n_groups).map(lambda x: False)

    if entry_agg_group_membership is not None and "freq_meta" not in group_globals:
        raise ValueError(
            "The 'freq_meta' global annotation must be supplied when the"
            " 'entry_agg_group_membership' is specified."
        )

    entry_agg_group_membership = entry_agg_group_membership or {}
    entry_agg_group_membership = {
        ann: [group_globals["freq_meta"].index(s) for s in strata]
        for ann, strata in entry_agg_group_membership.items()
    }

    n_adj_groups = hl.eval(hl.len(global_expr["adj_groups"]))
    if n_adj_groups != n_groups:
        raise ValueError(
            f"The number of elements in the 'adj_groups' ({n_adj_groups}) global "
            "annotation does not match the number of elements in the "
            f"'group_membership' annotation ({n_groups})!",
        )

    # Keep only the entries needed for the aggregation functions.
    select_expr = {**{ann: f[0](mt) for ann, f in entry_agg_funcs.items()}}
    has_adj = hl.eval(hl.any(global_expr["adj_groups"]))
    if has_adj:
        select_expr["adj"] = mt.adj

    mt = mt.select_entries(**select_expr)

    # Convert MT to HT with a row annotation that is an array of all samples entries
    # for that variant.
    ht = mt.localize_entries("entries", "cols")

    # For each stratification group in group_membership, determine the indices of the
    # samples that belong to that group.
    global_expr["indices_by_group"] = hl.range(n_groups).map(
        lambda g_i: hl.range(mt.count_cols()).filter(
            lambda s_i: ht.cols[s_i].group_membership[g_i]
        )
    )
    ht = ht.annotate_globals(**global_expr)

    # Pull out each annotation that will be used in the array aggregation below as its
    # own ArrayExpression. This is important to prevent memory issues when performing
    # the below array aggregations.
    ht = ht.select(
        *select_fields,
        **{ann: ht.entries.map(lambda e: e[ann]) for ann in select_expr.keys()},
    )

    def _agg_by_group(
        indices_by_group_expr: hl.expr.ArrayExpression,
        adj_groups_expr: hl.expr.ArrayExpression,
        agg_func: Callable,
        ann_expr: hl.expr.ArrayExpression,
    ) -> hl.expr.ArrayExpression:
        """
        Aggregate `agg_expr` by group using the `agg_func` function.

        :param indices_by_group_expr: ArrayExpression of indices of samples in each group.
        :param adj_groups_expr: ArrayExpression indicating whether each group is adj.
        :param agg_func: Aggregation function to apply to `ann_expr`.
        :param ann_expr: Expression to aggregate by group.
        :return: Aggregated array expression.
        """
        f_no_adj = lambda i, *args: agg_func(ann_expr[i])
        if has_adj:
            f = lambda i, adj: hl.if_else(
                adj, hl.agg.filter(ht.adj[i], f_no_adj(i)), f_no_adj(i)
            )
        else:
            f = f_no_adj

        return hl.map(
            lambda s_indices, adj: s_indices.aggregate(lambda i: f(i, adj)),
            indices_by_group_expr,
            adj_groups_expr,
        )

    # Add annotations for any supplied entry transform and aggregation functions.
    # Filter groups to only those in entry_agg_group_membership if specified.
    # If there are no specific entry group indices for an annotation, use ht[g]
    # to consider all groups without filtering.
    ht = ht.select(
        *select_fields,
        **{
            ann: _agg_by_group(
                *[
                    [ht[g][i] for i in entry_agg_group_membership.get(ann, [])] or ht[g]
                    for g in ["indices_by_group", "adj_groups"]
                ],
                agg_func=f[1],
                ann_expr=ht[ann],
            )
            for ann, f in entry_agg_funcs.items()
        },
    )

    return ht.drop("cols")


def update_structured_annotations(
    ht: hl.Table,
    annotation_update_exprs: Dict[str, hl.Expression],
    annotation_update_label: Optional[str] = None,
) -> hl.Table:
    """
    Update highly structured annotations on a Table.

    This function recursively updates annotations defined by `annotation_update_exprs`
    and if `annotation_update_label` is supplied, it checks if the sample annotations
    are different from the input and adds a flag to the Table, indicating which
    annotations have been updated for each sample.

    :param ht: Input Table with structured annotations to update.
    :param annotation_update_exprs: Dictionary of annotations to update, structured as
        they are structured on the input `ht`.
    :param annotation_update_label: Optional string of the label to use for an
        annotation indicating which annotations have been updated. Default is None, so
        no annotation is added.
    :return: Table with updated annotations and optionally a flag indicating which
        annotations were changed.
    """

    def _update_struct(
        struct_expr: hl.expr.StructExpression,
        update_exprs: Union[Dict[str, hl.expr.Expression], hl.expr.Expression],
    ) -> Tuple[Dict[str, hl.expr.BooleanExpression], Any]:
        """
        Update a StructExpression.

        :param struct_expr: StructExpression to update.
        :param update_exprs: Dictionary of annotations to update.
        :return: Tuple of the updated annotations and the updated flag.
        """
        if isinstance(update_exprs, dict):
            updated_struct_expr = {}
            updated_flag_expr = {}
            for ann, expr in update_exprs.items():
                if ann in struct_expr:
                    updated_flag, updated_ann = _update_struct(struct_expr[ann], expr)
                else:
                    updated_flag = {"": True}
                    updated_ann = expr
                updated_flag_expr.update(
                    {ann + ("." + k if k else ""): v for k, v in updated_flag.items()}
                )
                updated_struct_expr[ann] = updated_ann
            return updated_flag_expr, struct_expr.annotate(**updated_struct_expr)
        else:
            return {"": update_exprs != struct_expr}, update_exprs

    annotation_update_flag, updated_rows = _update_struct(
        ht.row_value, annotation_update_exprs
    )
    if annotation_update_label is not None:
        updated_rows = updated_rows.annotate(
            **{
                annotation_update_label: filter_utils.add_filters_expr(
                    filters=annotation_update_flag
                )
            }
        )

    return ht.annotate(**updated_rows)


def fill_missing_key_combinations(
    ht: hl.Table,
    fill_values: Dict[str, hl.expr.Expression],
    key_values: Optional[Dict[str, List]] = None,
) -> hl.Table:
    """
    Fill missing key combinations with requested fill values.

    This function fills missing key combinations in the input Table with requested fill
    values. The fill values are specified in the `fill_values` dictionary. The unique
    values for each key field are collected from the input Table unless specified in
    the `key_values` dictionary.

    This is useful when you want to ensure that all possible key combinations are
    present in a Table, even if they are missing from the input Table. This can
    happen when you are aggregating data and want to ensure that all possible key
    combinations are present in the output Table, not only the ones that are present.

    Example::

        .. code-block:: python

            ht = hl.Table.parallelize(
                [
                    {'key1': 'A', 'key2': 1, 'value': 10},
                    {'key1': 'A', 'key2': 2, 'value': 20},
                    {'key1': 'B', 'key2': 1, 'value': 30},
                ],
                hl.tstruct(key1=hl.tstr, key2=hl.tint32, value=hl.tint32),
                key=['key1', 'key2'],
            )
            fill_values = {'value': hl.missing(hl.tint32)}
            filled_ht = fill_missing_key_combinations(ht, fill_values)
            filled_ht.show()

        +------+------+
        | key1 | key2 | value |
        +------+------+
        |  A   |  1   |  10   |
        |  A   |  2   |  20   |
        |  B   |  1   |  30   |
        |  B   |  2   | null  |
        +------+------+

    In this example, the input table is missing the combination (B, 2).
    After applying `fill_missing_key_combinations`, the missing key combination
    (B, 2) is filled with the specified fill value for 'value' (null in this case).

    :param ht: Input Table containing key fields.
    :param fill_values: Dictionary of fill values to use for missing key combinations.
    :param key_values: Optional dictionary of unique values to use for each key field.
        Default is None. If None, the unique values for each key field are collected
        from the input Table.
    :return: Table with missing key combinations filled with requested fill values.
    """
    key_fields = list(ht.key.keys())

    # Extract unique values for each annotation.
    key_values = key_values or {}
    key_values = [
        key_values.get(f, ht.aggregate(hl.agg.collect_as_set(ht[f])))
        for f in key_fields
    ]

    # Create all combinations of the unique values for each key.
    all_combinations = list(itertools.product(*key_values))

    # Convert the list of combinations to a Hail Table.
    all_combinations_ht = hl.Table.parallelize(
        [{f: c for f, c in zip(key_fields, combo)} for combo in all_combinations],
        hl.tstruct(**{f: ht[f].dtype for f in key_fields}),
    ).key_by(*key_fields)

    # Left join original table with all_combinations to include all possible
    # combinations.
    ht = all_combinations_ht.join(ht, how="left")

    # Fill missing row annotations with requested fill values.
    ht = ht.annotate(**{f: hl.coalesce(ht[f], v) for f, v in fill_values.items()})

    return ht


def missing_struct_expr(
    dtypes: hl.expr.types.tstruct,
) -> hl.expr.StructExpression:
    """
    Create a struct of missing values corresponding to each field and type in the input struct.

    :param dtypes: StructExpression containing the field names and missing values.
    """
    return hl.struct(**{f: hl.missing(t) for f, t in dtypes.items()})


def add_gks_vrs(
    input_locus: hl.locus,
    input_vrs: hl.struct,
) -> dict:
    """
    Generate a dictionary containing VRS information from a given locus and struct of VRS information.

    Dict will have GA4GH GKS VRS structure.

    :param input_locus: Locus field from a struct (locus of result of running .collect() on a Hail table).
    :param input_vrs: VRS struct (such as from a ht.info.vrs field).
    :return: Python dictionary conforming to GA4GH GKS VRS structure.
    """
    # NOTE: The pinned ga4gh.vrs module breaks logging when this annotations module is
    # imported. Importing ga4gh here to avoid this issue.
    import ga4gh.core as ga4gh_core
    import ga4gh.vrs as ga4gh_vrs

    build_in = input_locus.reference_genome.name
    chr_in = input_locus.contig

    chrom_dict = VRS_CHROM_IDS[build_in]
    vrs_id = input_vrs.VRS_Allele_IDs[1]
    vrs_chrom_id = chrom_dict[chr_in]
    vrs_start_value = input_vrs.VRS_Starts[1]
    vrs_end_value = input_vrs.VRS_Ends[1]
    vrs_state_sequence = input_vrs.VRS_States[1]

    vrs_dict_out = {
        "_id": vrs_id,
        "type": "Allele",
        "location": {
            "type": "SequenceLocation",
            "sequence_id": vrs_chrom_id,
            "interval": {
                "start": {"type": "Number", "value": vrs_start_value},
                "end": {"type": "Number", "value": vrs_end_value},
                "type": "SequenceInterval",
            },
        },
        "state": {"type": "LiteralSequenceExpression", "sequence": vrs_state_sequence},
    }

    location_id = ga4gh_core._internal.identifiers.ga4gh_identify(
        ga4gh_vrs.models.SequenceLocation(**vrs_dict_out["location"])
    )

    vrs_dict_out["location"]["_id"] = location_id

    return vrs_dict_out


def add_gks_va(
    input_struct: hl.struct,
    label_name: str = "gnomAD",
    label_version: str = "3.1.2",
    ancestry_groups: list = None,
    ancestry_groups_dict: dict = None,
    by_sex: bool = False,
    freq_index_dict: dict = None,
) -> dict:
    """
    Generate Python dictionary containing GKS VA annotations.

    Populate the dictionary with frequency information conforming to the GKS VA frequency schema.
    If ancestry_groups or by_sex is provided, also include subcohort schemas for each cohort.
    If input_struct has mean_depth, it is added to ancillaryResults.
    This annotation is added under the gks_va_freq_dict field of the table.
    The focusAllele field is not populated, and must be filled in by the caller.

    :param input_struct: Hail struct for a desired variant (such as result of running .collect()[0] on a Table).
    :param label_name: Label name to use within the returned dictionary. Example: "gnomAD".
    :param label_version: String listing the version of the table being used. Example: "3.1.2".
    :param ancestry_groups: List of strings of shortened names of cohorts to return results for.
        Example: ['afr','fin','nfe']. Default is None.
    :param ancestry_groups_dict: Dict mapping shortened genetic ancestry group names to full names.
        Example: {'afr':'African/African American'}. Default is None.
    :param by_sex: Boolean to include breakdown of cohorts by inferred sex (XX and XY) as well.
        Default is None.
    :freq_index_dict: Dict mapping groups to their index for freq info in ht.freq_index_dict[0].
        Default is None.
    :return: Tuple containing a dictionary containing GKS VA frequency information,
        (split by ancestry groups and sex if desired) for the specified variant.
    """
    # Throw warnings if contradictory arguments passed.
    if by_sex and not ancestry_groups:
        logger.warning(
            "Splitting whole database by sex is not yet supported. If using 'by_sex',"
            " please also specify 'ancestry_groups' to stratify by."
        )

    contig = input_struct.locus.contig
    pos = input_struct.locus.position
    ref = input_struct.alleles[0]
    var = input_struct.alleles[1]
    gnomad_id = f"{contig}-{pos}-{ref}-{var}"

    # Define function to return a frequency report dictionary for a given group
    def _create_group_dicts(
        group_id: str,
        group_label: str,
        group_sex: str = None,
    ) -> dict:
        """
        Generate a dictionary containing the frequency information of a given variant for a given group.

        :param group_index: Index of frequency within the 'freq' annotation for the desired group.
        :param group_id: String containing variant, genetic ancestry group, and sex (if requested).
            - Example: "chr19-41094895-C-T.afr.XX".
        :param group_label: String containing the full name of genetic ancestry group requested.
            - Example: "African/African American".
        :param group_sex: String indicating the sex of the group.
            - Example: "XX" or "XY".
        :return: Dictionary containing variant frequency information,
            - (by genetic ancestry group and sex if desired) for specified variant.
        """
        if group_sex:
            cohort_id = f"{group_id.upper()}.{group_sex}"
            freq_index_key = f"{group_id}_{group_sex}_adj"
        else:
            cohort_id = f"{group_id.upper()}"
            freq_index_key = f"{group_id}_adj"
        record_id = f"{gnomad_id}.{cohort_id}"

        # Obtain frequency information for the specified variant.
        group_freq = input_struct.freq[freq_index_dict[freq_index_key]]

        # Cohort characteristics.
        characteristics = []
        characteristics.append({"name": "genetic ancestry", "value": group_label})
        if group_sex is not None:
            characteristics.append({"name": "biological sex", "value": group_sex})

        # Dictionary to be returned containing information for a specified group.
        freq_record = {
            "id": record_id,
            "type": "CohortAlleleFrequency",
            "label": f"{group_label} Cohort Allele Frequency for {gnomad_id}",
            "focusAllele": "#/focusAllele",
            "focusAlleleCount": group_freq["AC"],
            "locusAlleleCount": group_freq["AN"],
            "alleleFrequency": (
                group_freq["AF"] if group_freq["AF"] is not None else 0.0
            ),
            "cohort": {"id": cohort_id, "characteristics": characteristics},
            "ancillaryResults": {"homozygotes": group_freq["homozygote_count"]},
        }

        # Add hemizygote allele count if variant is non-autosomal/non-PAR.
        # Only XY groups can be hemizygous. Other group AC is mixed homo/hetero.
        # If not a by_sex group, include the XY hemizygote count for XY subgroup.
        if not input_struct.in_autosome_or_par:
            if group_sex == "XY":
                freq_record["ancillaryResults"]["hemizygotes"] = group_freq.AC
            elif group_sex is None:
                # Group is not by_sex, but still need to report hemizygotes.
                hemi_group_freq = input_struct.freq[
                    freq_index_dict[f"{group_id}_XY_adj"]
                ]
                freq_record["ancillaryResults"]["hemizygotes"] = hemi_group_freq.AC

        return freq_record

    # Create a list to then add the dictionaries for frequency reports for
    # different ancestry groups to.
    list_of_group_info_dicts = []

    # Iterate through provided groups and generate dictionaries.
    if ancestry_groups:
        for group in ancestry_groups:
            group_result = _create_group_dicts(
                group_id=group,
                group_label=ancestry_groups_dict[group],
            )

            # If specified, stratify group information by sex.
            if by_sex:
                sex_list = []
                for sex in ["XX", "XY"]:
                    sex_result = _create_group_dicts(
                        group_id=group,
                        group_label=ancestry_groups_dict[group],
                        group_sex=sex,
                    )
                    sex_list.append(sex_result)

                group_result["subcohortFrequency"] = sex_list

            list_of_group_info_dicts.append(group_result)

    # Add overall frequency, via label 'adj' which is currently stored at
    # position #1 (index 0).
    overall_freq = input_struct.freq[0]

    # Create final dictionary to be returned.
    final_freq_dict = {
        "id": f"{label_name}-{label_version}-{gnomad_id}",
        "type": "CohortAlleleFrequency",
        "label": f"Overall Cohort Allele Frequency for {gnomad_id}",
        "derivedFrom": {
            "id": f"{label_name}{label_version}",
            "type": "DataSet",
            "label": f"{label_name} v{label_version}",
            "version": f"{label_version}",
        },
        "focusAllele": (
            ""
        ),  # Information can be populated with the result of add_gks_vrs()
        "focusAlleleCount": overall_freq["AC"],
        "locusAlleleCount": overall_freq["AN"],
        "alleleFrequency": (
            overall_freq["AF"] if overall_freq["AF"] is not None else 0.0
        ),
        "cohort": {"id": "ALL"},
    }

    # Create ancillaryResults for additional frequency and popMaxFAF95 information.
    ancillaryResults = {
        "homozygotes": overall_freq["homozygote_count"],
    }

    # Add hemizygote count if not autosomal or PAR.
    if not input_struct.in_autosome_or_par:
        hemizygote_count = input_struct.freq[freq_index_dict["XY_adj"]].AC
        ancillaryResults["hemizygotes"] = hemizygote_count

    # Add group max FAF if it exists
    if input_struct.grpMaxFAF95.popmax_population is not None:
        ancillaryResults["grpMaxFAF95"] = {
            "frequency": input_struct.grpMaxFAF95.popmax,
            "confidenceInterval": 0.95,
            "groupId": (
                f"{gnomad_id}.{input_struct.grpMaxFAF95.popmax_population.upper()}"
            ),
        }

    # Add joint group max FAF if it exists.
    if (
        "jointGrpMaxFAF95" in input_struct
        and input_struct.jointGrpMaxFAF95.popmax_population is not None
    ):
        ancillaryResults["jointGrpMaxFAF95"] = {
            "frequency": input_struct.jointGrpMaxFAF95.popmax,
            "confidenceInterval": 0.95,
            "groupId": (
                f"{gnomad_id}.{input_struct.jointGrpMaxFAF95.popmax_population.upper()}"
            ),
        }

    final_freq_dict["ancillaryResults"] = ancillaryResults

    # Check allele balance for heterozygotes values.
    # Flagged allele balance values are those in bins > 0.90.
    # Each bin is 0.05, so flagged values are in the last 2 bins.
    if len(input_struct.ab_hist_alt.bin_freq) != 20:
        raise ValueError(
            f"{gnomad_id} ab_hist_alt.bin_freq had "
            f"{len(input_struct.ab_hist_alt.bin_freq)} items, expected 20"
        )
    # The bin_freq should be in order but we can verify the order from bin_edges.
    ab_bin_freq = list(
        map(
            lambda x: x[1],
            sorted(
                zip(
                    input_struct.ab_hist_alt.bin_edges,
                    input_struct.ab_hist_alt.bin_freq,
                ),
                key=lambda x: x[0],
            ),
        )
    )

    qualityMeasures = {
        "qcFilters": list(input_struct.filters),
        "lowComplexityRegion": input_struct.lcr,
        "heterozygousSkewedAlleleCount": sum(ab_bin_freq[-2:]),
    }

    # Add coverage depth statistics if the input was annotated
    # with coverage information.
    if "mean_depth" in input_struct:
        qualityMeasures["meanDepth"] = input_struct.mean_depth

    if "fraction_cov_over_20" in input_struct:
        qualityMeasures["fractionCoverage20x"] = input_struct.fraction_cov_over_20

    # Add monoallelic flag (all samples homozygous for alternate allele)
    qualityMeasures["monoallelic"] = input_struct.monoallelic

    final_freq_dict["qualityMeasures"] = qualityMeasures

    # If ancestry_groups were passed, add the ancestry group dictionary to the
    # final frequency dictionary to be returned.
    if ancestry_groups:
        final_freq_dict["subcohortFrequency"] = list_of_group_info_dicts

    return final_freq_dict


def get_copy_state_by_sex(
    locus_expr: hl.expr.LocusExpression,
    is_xx_expr: hl.expr.BooleanExpression,
) -> Tuple[
    hl.expr.BooleanExpression, hl.expr.BooleanExpression, hl.expr.BooleanExpression
]:
    """
    Determine the copy state of a variant by its locus and the sex karotype of a sample.

    This function assumes that the sample contains only XX and XY karyotypes. It does
    not account for ambiguous sex or aneuploidies (e.g., XXY, XYY).

    :param locus_expr: LocusExpression of the variant.
    :param is_xx_expr: BooleanExpression indicating whether the sample has an XX sex
       karyotype.
    :return: Tuple of BooleanExpressions:
        - diploid_expr: True if the variant is in autosomes or PAR regions, or in the X non-PAR region for XX individuals.
        - hemi_x_expr: True if the variant is in the X non-PAR region for XY individuals.
        - hemi_y_expr: True if the variant is in the Y non-PAR region for XY individuals.
    """
    diploid_expr = locus_expr.in_autosome_or_par() | (
        locus_expr.in_x_nonpar() & is_xx_expr
    )
    hemi_x_expr = locus_expr.in_x_nonpar() & ~is_xx_expr
    hemi_y_expr = locus_expr.in_y_nonpar() & ~is_xx_expr
    return diploid_expr, hemi_x_expr, hemi_y_expr

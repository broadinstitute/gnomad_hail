import numpy as np
from .generic import *
from gnomad_hail.resources import exome_calling_intervals_path
from .gnomad_functions import logger, filter_to_adj


def filter_rows_for_qc(
        mt: hl.MatrixTable,
        min_af: Optional[float] = 0.001,
        min_callrate: Optional[float] = 0.99,
        min_inbreeding_coeff_threshold: Optional[float] = -0.8,
        min_hardy_weinberg_threshold: Optional[float] = 1e-8,
        apply_hard_filters: bool = True,
        bi_allelic_only: bool = True,
        snv_only: bool = True
) -> hl.MatrixTable:
    """
    Annotates rows with `sites_callrate`, `site_inbreeding_coeff` and `af`, then applies thresholds.
    AF and callrate thresholds are taken from gnomAD QC; inbreeding coeff, MQ, FS and QD filters are taken from GATK best practices

    Note
    ----
    This function expect the typical ``info`` annotation of type struct with fields ``MQ``, ``FS`` and ``QD``
    if applying hard filters.

    :param MatrixTable mt: Input MT
    :param float min_af: Minimum site AF to keep. Not applied if set to ``None``.
    :param float min_callrate: Minimum site call rate to keep. Not applied if set to ``None``.
    :param float min_inbreeding_coeff_threshold: Minimum site inbreeding coefficient to keep. Not applied if set to ``None``.
    :param float min_hardy_weinberg_threshold: Minimum site HW test p-value to keep. Not applied if set to ``None``.
    :param bool apply_hard_filters: Whether to apply standard GAKT default site hard filters: QD >= 2, FS <= 60 and MQ >= 30
    :param bool bi_allelic_only: Whether to only keep bi-allelic sites or include multi-allelic sites too
    :param bool snv_only: Whether to only keep SNVs or include other variant types
    :return: annotated and filtered table
    :rtype: MatrixTable
    """
    annotation_expr = {}

    if min_af is not None:
        annotation_expr['af'] = hl.agg.mean(mt.GT.n_alt_alleles()) / 2
    if min_callrate is not None:
        annotation_expr['site_callrate'] = hl.agg.fraction(hl.is_defined(mt.GT))
    if min_inbreeding_coeff_threshold is not None:
        annotation_expr['site_inbreeding_coeff'] = bi_allelic_site_inbreeding_expr(mt.GT)
    if min_hardy_weinberg_threshold is not None:
        annotation_expr['hwe'] = hl.agg.hardy_weinberg_test(mt.GT)

    if annotation_expr:
        mt = mt.annotate_rows(**annotation_expr)

    filter_expr = []
    if min_af is not None:
        filter_expr.append((mt.af > min_af))
    if min_callrate is not None:
        filter_expr.append((mt.site_callrate > min_callrate))
    if min_inbreeding_coeff_threshold is not None:
        filter_expr.append((mt.site_inbreeding_coeff > min_inbreeding_coeff_threshold))
    if min_hardy_weinberg_threshold is not None:
        filter_expr.append((mt.hwe.p_value > min_hardy_weinberg_threshold))
    if snv_only:
        filter_expr.append(hl.is_snp(mt.alleles[0], mt.alleles[1]))
    if bi_allelic_only:
        filter_expr.append(bi_allelic_expr(mt))

    if apply_hard_filters:
        if 'info' in mt.row_value:
            if 'QD' in mt.info:
                filter_expr.append((mt.info.QD >= 2))
            else:
                logger.warn("Could not apply QD hard filter, as `info.QD` not found in schema.")
            if 'FS' in mt.info:
                filter_expr.append((mt.info.FS <= 60))
            else:
                logger.warn("Could not apply FS hard filter, as `info.FS` not found in schema.")
            if 'MQ' in mt.info:
                filter_expr.append((mt.info.MQ >= 30))
            else:
                logger.warn("Could not apply MQ hard filter, as `info.MQ` not found in schema.")
        else:
            logger.warn("Could not apply hard filters as `info` not found in schema.")

    return mt.filter_rows(functools.reduce(operator.iand, filter_expr))


def get_qc_mt(
        mt: hl.MatrixTable,
        adj_only: bool = True,
        min_af: float = 0.001,
        min_callrate: float = 0.99,
        min_inbreeding_coeff_threshold: float = -0.8,
        min_hardy_weinberg_threshold: Optional[float] = 1e-8,
        apply_hard_filters: bool = True,
        ld_r2: Optional[float] = 0.1,
        filter_lcr: bool = True,
        filter_decoy: bool = True,
        filter_segdup: bool = True,
        filter_exome_low_coverage_regions: bool = False,
        high_conf_regions: Optional[List[str]] = None
) -> hl.MatrixTable:
    """
    Creates a QC-ready MT by keeping:
    - Variants outside known problematic regions
    - Bi-allelic SNVs only
    - Variants passing hard thresholds
    - Variants passing the set call rate and MAF thresholds
    - Genotypes passing on gnomAD ADJ criteria (GQ>=20, DP>=10, AB>0.2 for hets)

    In addition, the MT will be LD-pruned if `ld_r2` is set.

    :param MatrixTable mt: Input MT
    :param bool adj_only: If set, only ADJ genotypes are kept. This filter is applied before the call rate and AF calculation.
    :param float min_af: Minimum allele frequency to keep
    :param float min_callrate: Minimum call rate to keep
    :param float min_inbreeding_coeff_threshold: Minimum site inbreeding coefficient to keep
    :param float min_hardy_weinberg_threshold: Minimum site HW test p-value to keep
    :param bool apply_hard_filters: Whether to apply standard GAKT default site hard filters: QD >= 2, FS <= 60 and MQ >= 30
    :param float ld_r2: Minimum r2 to keep when LD-pruning (set to `None` for no LD pruning)
    :param bool filter_lcr: Filter LCR regions
    :param bool filter_decoy: Filter decoy regions
    :param bool filter_segdup: Filter segmental duplication regions
    :param bool filter_exome_low_coverage_regions: If set, only high coverage exome regions (computed from gnomAD are kept)
    :param list of str high_conf_regions: If given, the data will be filtered to only include variants in those regions
    :return: Filtered MT
    :rtype: MatrixTable
    """
    logger.info("Creating QC MatrixTable")
    if ld_r2 is not None:
        logger.warn("The LD-prune step of this function requires non-preemptible workers only!")

    qc_mt = filter_low_conf_regions(
        mt,
        filter_lcr=filter_lcr,
        filter_decoy=filter_decoy,
        filter_segdup=filter_segdup,
        filter_exome_low_coverage_regions=filter_exome_low_coverage_regions, high_conf_regions=high_conf_regions
    )

    if adj_only:
        qc_mt = filter_to_adj(qc_mt)  # TODO: Make sure that this works fine before call rate filtering

    qc_mt = filter_rows_for_qc(
        qc_mt,
        min_af,
        min_callrate,
        min_inbreeding_coeff_threshold,
        min_hardy_weinberg_threshold,
        apply_hard_filters
    )

    if ld_r2 is not None:
        qc_mt = qc_mt.persist()
        unfiltered_qc_mt = qc_mt.unfilter_entries()
        pruned_ht = hl.ld_prune(unfiltered_qc_mt.GT, r2=ld_r2)
        qc_mt = qc_mt.filter_rows(hl.is_defined(pruned_ht[qc_mt.row_key]))

    qc_mt = qc_mt.annotate_globals(
        qc_mt_params=hl.struct(
            adj_only=adj_only,
            min_af=min_af,
            min_callrate=min_callrate,
            inbreeding_coeff_threshold=min_inbreeding_coeff_threshold,
            apply_hard_filters=apply_hard_filters,
            ld_r2=ld_r2 if ld_r2 is not None else hl.null(hl.tfloat32),
            filter_exome_low_coverage_regions=filter_exome_low_coverage_regions,
            high_conf_regions=high_conf_regions if high_conf_regions is not None else hl.null(hl.tarray(hl.tstr))
        )
    )
    return qc_mt.annotate_cols(sample_callrate=hl.agg.fraction(hl.is_defined(qc_mt.GT)))


def compute_callrate_mt(
        mt: hl.MatrixTable,
        intervals_ht: hl.Table,
        bi_allelic_only: bool = True,
        autosomes_only: bool = True,
        match: bool = True,
) -> hl.MatrixTable:
    """
    Computes a sample/interval MT with each entry containing the call rate for that sample/interval.
    This can be used as input for imputing exome sequencing platforms.

    Note
    ----
    The input interval HT should have a key of type Interval.
    The resulting table will have a key of the same type as the `intervals_ht` table and
    contain an `interval_info` field containing all non-key fields of the `intervals_ht`.

    :param MatrixTable mt: Input MT
    :param Table intervals_ht: Table containing the intervals. This table has to be keyed by locus.
    :param bool bi_allelic_only: If set, only bi-allelic sites are used for the computation
    :param bool autosomes_only: If set, only autosomal intervals are used.
    :param bool matches: If set, returns all intervals in intervals_ht that overlap the locus in the input MT.
    :return: Callrate MT
    :rtype: MatrixTable
    """
    logger.info('Computing call rate MatrixTable')

    if len(intervals_ht.key) != 1 or not isinstance(intervals_ht.key[0], hl.expr.IntervalExpression):
        logger.warn(f'Call rate matrix computation expects `intervals_ht` with a key of type Interval. Found: {intervals_ht.key}')

    if autosomes_only:
        callrate_mt = filter_to_autosomes(mt)

    if bi_allelic_only:
        callrate_mt = callrate_mt.filter_rows(bi_allelic_expr(callrate_mt))

    intervals_ht = intervals_ht.annotate(_interval_key=intervals_ht.key)
    callrate_mt = callrate_mt.annotate_rows(_interval_key=intervals_ht.index(callrate_mt.locus, all_matches=match)._interval_key)

    if match:
        callrate_mt = callrate_mt.explode_rows('_interval_key')

    callrate_mt = callrate_mt.filter_rows(hl.is_defined(callrate_mt._interval_key.interval))
    callrate_mt = callrate_mt.select_entries(GT=hl.or_missing(hl.is_defined(callrate_mt.GT), hl.struct()))
    callrate_mt = callrate_mt.group_rows_by(**callrate_mt._interval_key).aggregate(callrate=hl.agg.fraction(hl.is_defined(callrate_mt.GT)))
    intervals_ht = intervals_ht.drop('_interval_key')
    callrate_mt = callrate_mt.annotate_rows(interval_info=hl.struct(**intervals_ht[callrate_mt.row_key]))
    return callrate_mt


def run_platform_pca(
        callrate_mt: hl.MatrixTable,
        binarization_threshold: Optional[float] = 0.25
) -> Tuple[List[float], hl.Table, hl.Table]:
    """
    Runs a PCA on a sample/interval MT with each entry containing the call rate.
    When `binzarization_threshold` is set, the callrate is transformed to a 0/1 value based on the threshold.
    E.g. with the default threshold of 0.25, all entries with a callrate < 0.25 are considered as 0s, others as 1s.

    :param MatrixTable callrate_mt: Input callrate MT
    :param float binarization_threshold: binzarization_threshold. None is no threshold desired
    :return: eigenvalues, scores_ht, loadings_ht
    :rtype: (list of float, Table Table)
    """
    logger.info("Running platform PCA")

    if binarization_threshold is not None:
        callrate_mt = callrate_mt.annotate_entries(callrate=hl.int(callrate_mt.callrate > binarization_threshold))
    # Center until Hail's PCA does it for you
    callrate_mt = callrate_mt.annotate_rows(mean_callrate=hl.agg.mean(callrate_mt.callrate))
    callrate_mt = callrate_mt.annotate_entries(callrate=callrate_mt.callrate - callrate_mt.mean_callrate)
    eigenvalues, scores, loadings = hl.pca(callrate_mt.callrate, compute_loadings=True)  # TODO:  Evaluate whether computing loadings is a good / worthy thing
    logger.info('Platform PCA eigenvalues: {}'.format(eigenvalues))

    return eigenvalues, scores, loadings


def assign_platform_from_pcs(
        platform_pca_scores_ht: hl.Table,
        pc_scores_ann: str = 'scores',
        hdbscan_min_cluster_size: Optional[int] = None,
        hdbscan_min_samples: int = None
) -> hl.Table:
    """
    Assigns platforms using HBDSCAN on the results of call rate PCA.

    :param Table platform_pca_scores_ht: Input table with the PCA score for each sample
    :param str pc_scores_ann: Field containing the scores
    :param int hdbscan_min_cluster_size: HDBSCAN `min_cluster_size` parameter. If not specified the smallest of 500 and 0.1*n_samples will be used.
    :param int hdbscan_min_samples: HDBSCAN `min_samples` parameter
    :return: A Table with a `qc_platform` annotation containing the platform based on HDBSCAN clustering
    """
    import hdbscan
    logger.info("Assigning platforms based on platform PCA clustering")

    # Read and format data for clustering
    data = platform_pca_scores_ht.to_pandas()
    callrate_data = np.matrix(data[pc_scores_ann].tolist())
    logger.info('Assigning platforms to {} samples.'.format(len(callrate_data)))

    # Cluster data
    if hdbscan_min_cluster_size is None:
        hdbscan_min_cluster_size = min(500, 0.1 * data.shape[0])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples)
    cluster_labels = clusterer.fit_predict(callrate_data)
    n_clusters = len(set(cluster_labels)) - (-1 in cluster_labels)  # NOTE: -1 is the label for noisy (un-classifiable) data points
    logger.info('Found {} unique platforms during platform imputation.'.format(n_clusters))

    data['qc_platform'] = cluster_labels
    ht = hl.Table.from_pandas(data, key=[*platform_pca_scores_ht.key])
    ht = ht.annotate(qc_platform='platform_' + hl.str(ht.qc_platform))
    return ht


# TODO: This should be reviewed / merged with work from Kristen
def infer_sex(
        x_mt: hl.MatrixTable,  # TODO: This feels somewhat unsatisfying to provide two MTs. Maybe just reapply the QC MT filters to both (minus callrate for Y)?
        y_mt: hl.MatrixTable,
        platform_ht: hl.Table,
        male_min_f_stat: float,
        female_max_f_stat: float,
        min_male_y_sites_called: int = 500,
        max_y_female_call_rate: float = 0.15,
        min_y_male_call_rate: float = 0.8
) -> hl.Table:
    """
    Imputes sample sex based on X-chromosome heterozygosity and Y-callrate (if Y calls are present)

    :param x_mt:
    :param y_mt:
    :param platform_ht:
    :param male_min_f_stat:
    :param female_max_f_stat:
    :param min_male_y_sites_called:
    :param max_y_female_call_rate:
    :param min_y_male_call_rate:
    :return:
    """
    logger.info("Imputing samples sex")

    x_mt = hl.filter_intervals(x_mt, [hl.parse_locus_interval(x_contig) for x_contig in get_reference_genome(x_mt.locus).x_contigs])
    x_ht = hl.impute_sex(x_mt.GT, aaf_threshold=0.05, female_threshold=female_max_f_stat, male_threshold=male_min_f_stat)
    y_mt = hl.filter_intervals(y_mt, [hl.parse_locus_interval(y_contig) for y_contig in get_reference_genome(y_mt.locus).y_contigs])
    y_mt = y_mt.filter_rows(y_mt.locus.in_y_nonpar())
    sex_ht = y_mt.annotate_cols(
        qc_platform=platform_ht[y_mt.col_key].qc_platform,
        is_female=x_ht[y_mt.col_key].is_female,
        y_call_rate=hl.agg.fraction(hl.is_defined(y_mt.GT)),
        n_y_sites_called=hl.agg.count_where(hl.is_defined(y_mt.GT)),
        **{f'x_{ann}': x_ht[y_mt.col_key][ann] for ann in x_ht.row_value if ann != 'is_female'}
    ).cols()

    mean_male_y_sites_called = sex_ht.aggregate(hl.agg.filter(~sex_ht.is_female, hl.agg.group_by(sex_ht.qc_platform, hl.agg.mean(sex_ht.n_y_sites_called))))
    y_call_rate_stats = sex_ht.aggregate(
        hl.agg.filter(
            hl.is_defined(sex_ht.is_female),
            hl.agg.group_by(hl.tuple([sex_ht.qc_platform, sex_ht.is_female]),
                            hl.agg.stats(sex_ht.y_call_rate)
                            )
        )
    )

    no_y_call_rate_platforms = set()
    for platform, sites_called in mean_male_y_sites_called.items():
        if sites_called < min_male_y_sites_called:
            logger.warn(f"Mean number of sites in males on Y chromosome for platform {platform} is < {min_male_y_sites_called} ({sites_called} sites found). Y call rate filter will NOT be applied for samples on platform {platform}.")
            no_y_call_rate_platforms.add(platform)

    sex_ht = sex_ht.annotate_globals(y_call_rate_stats=y_call_rate_stats,
                                     no_y_call_rate_platforms=no_y_call_rate_platforms if no_y_call_rate_platforms else hl.empty_set(platform_ht.qc_platform.dtype))
    y_female_stats = sex_ht.y_call_rate_stats[(sex_ht.qc_platform, True)]
    y_male_stats = sex_ht.y_call_rate_stats[(sex_ht.qc_platform, False)]
    sex_ht = sex_ht.annotate(
        is_female=(
            hl.case()
                .when(sex_ht.no_y_call_rate_platforms.contains(sex_ht.qc_platform), sex_ht.is_female)
                .when(sex_ht.is_female & ((sex_ht.y_call_rate - y_female_stats.min) / (y_male_stats.max - y_female_stats.min) < max_y_female_call_rate), True)
                .when(~sex_ht.is_female & ((sex_ht.y_call_rate - y_female_stats.min) / (y_male_stats.max - y_female_stats.min) > min_y_male_call_rate), False)
                .or_missing()
        )
    )
    sex_ht = sex_ht.annotate_globals(
        impute_sex_params=hl.struct(
            male_min_f_stat=male_min_f_stat,
            female_max_f_stat=female_max_f_stat,
            min_male_y_sites_called=min_male_y_sites_called,
            max_y_female_call_rate=max_y_female_call_rate,
            min_y_male_call_rate=min_y_male_call_rate
        )
    )

    sex_ht = sex_ht.drop('qc_platform')
    return (sex_ht)


def get_sex_expr(
    chr_x_ploidy: hl.expr.NumericExpression,
    chr_y_ploidy: hl.expr.NumericExpression,
    f_stat: hl.expr.NumericExpression,
    xx_x_ploidy_cutoffs: Tuple[float, float] = (1.4, 2.25),
    xy_x_ploidy_cutoffs: Tuple[float, float] = (0.5, 1.4),
    xx_y_ploidy_cutoff: float = 0.1,
    xy_y_ploidy_cutoffs: Tuple[float, float] = (0.15, 1.2),
    yy_y_ploidy_cutoff: float = 1.3,
    xxx_x_ploidy_cutoff:float = 2.5,
    f_stat_female_cutoff: float = -0.2,
    f_stat_male_cutoff: float = 0.2
) -> hl.expr.StructExpression:
    # TODO: Automate cutoffs
    # TODO: Provide alternatives in case of missing annotations (e.g. no chr_y_ploidy)
    """
    Creates a struct with the following annotations:
    - karyoptype (str)
    - sex (str), which can be either 'male', 'female' or missing
    - is_female (missing for karyotypes that aren't either 'XX' or 'XY')

    :param NumericExpression chr_x_ploidy: chrom X ploidy (or relative ploidy)
    :param NumericExpression  chr_y_ploidy: chrom Y ploidy (or relative ploidy)
    :param NumericExpression f_stat: chrom X F-stat
    :param Tuple[float, float] xx_x_ploidy_cutoffs: Boundaries around the chom X ploidy for females
    :param Tuple[float, float] xy_x_ploidy_cutoffs: Boundaries around the chom X ploidy for males
    :param float xx_y_ploidy_cutoff: Max y ploidy for females
    :param Tuple[float, float] xy_y_ploidy_cutoffs: Boundaries around the chom Y ploidy for males
    :param float yy_y_ploidy_cutoff: Min chrom Y ploidy for YY
    :param float xxx_x_ploidy_cutoff: Min chrom X ploidy for XXX
    :param float f_stat_female_cutoff: Min f-stat for females
    :param flat f_stat_male_cutoff: Max f-stat for males
    :return: Struct expression with sex anotations
    :rtype: StructExpression
    """

    sex_expr = hl.struct(
        sex_karyotype=(
            hl.case()
                .when(
                (chr_x_ploidy > xx_x_ploidy_cutoffs[0]) &
                (chr_x_ploidy < xx_x_ploidy_cutoffs[1]) &
                (chr_y_ploidy < xx_y_ploidy_cutoff) &
                (f_stat < f_stat_female_cutoff),
                'XX')
                .when(
                (chr_x_ploidy > xy_x_ploidy_cutoffs[0]) &
                (chr_x_ploidy < xy_x_ploidy_cutoffs[1]) &
                (chr_y_ploidy > xy_y_ploidy_cutoffs[0]) &
                (chr_y_ploidy < xy_y_ploidy_cutoffs[1]) &
                (f_stat > f_stat_male_cutoff),
                'XY')
                .when(
                (chr_x_ploidy > xy_x_ploidy_cutoffs[0]) &
                (chr_x_ploidy < xy_x_ploidy_cutoffs[1]) &
                (chr_y_ploidy > yy_y_ploidy_cutoff) &
                (f_stat > f_stat_male_cutoff),
                'XYY')
                .when(
                (chr_x_ploidy > xx_x_ploidy_cutoffs[0]) &
                (chr_x_ploidy < xx_x_ploidy_cutoffs[1]) &
                (chr_y_ploidy > xy_y_ploidy_cutoffs[0]) &
                (chr_y_ploidy < xy_y_ploidy_cutoffs[1]) &
                (f_stat < f_stat_female_cutoff),
                'XXY')
                .when(
                (chr_x_ploidy > xxx_x_ploidy_cutoff) &
                (chr_y_ploidy < xx_y_ploidy_cutoff) &
                (f_stat < f_stat_female_cutoff),
                'XXX')
                .when(
                (chr_y_ploidy < xx_y_ploidy_cutoff) &
                (chr_x_ploidy > xy_x_ploidy_cutoffs[0]) &
                (chr_x_ploidy < xy_x_ploidy_cutoffs[1]) &
                (f_stat > f_stat_male_cutoff),
                'X0')
                .default('Ambiguous')
        )
    )

    return sex_expr.annotate(
        sex=(
            hl.case()
                .when(sex_expr.sex_karyotype == 'XX', 'female')
                .when(sex_expr.sex_karyotype == 'XY', 'male')
                .or_missing()
        ),
        is_female=(
            hl.case()
                .when(sex_expr.sex_karyotype == 'XX', True)
                .when(sex_expr.sex_karyotype == 'XY', False)
                .or_missing()
        )
    )


def filter_duplicate_samples(
        relatedness_ht: hl.Table,
        samples_rankings_ht: hl.Table,
        rank_ann: str = 'rank'
):
    """
    Creates a HT with duplicated samples sets.
    Each row is indexed by the sample that is kept and also contains the set of duplicate samples that should be filtered.

    `samples_rankings_ht` is a HT containing a global rank for each of the sample (smaller is better).

    :param Table relatedness_ht: Input relatedness HT
    :param Table samples_rankings_ht: HT with global rank for each sample
    :param str rank_ann: Annotation in `samples_ranking_ht` containing each sample global rank (smaller is better).
    :return: HT with duplicate sample sets, including which to keep/filter
    :rtype: Table
    """
    logger.info("Getting duplicate samples")
    dups = get_duplicated_samples(relatedness_ht)
    logger.info(f"Found {len(dups)} duplicate sets.")
    dups_ht = hl.Table.parallelize([hl.struct(dup_set=i, dups=dups[i]) for i in range(0, len(dups))])
    dups_ht = dups_ht.explode(dups_ht.dups, name='_dup')
    if isinstance(dups_ht._dup, hl.expr.StructExpression):
        dups_ht = dups_ht.key_by(**dups_ht._dup)
    else:
        dups_ht = dups_ht.key_by('_dup')
    dups_ht = dups_ht.annotate(rank=samples_rankings_ht[dups_ht.key][rank_ann])
    dups_cols = hl.bind(
        lambda x: hl.struct(
            kept=x[0],
            filtered=x[1:]
        ),
        hl.sorted(hl.agg.collect(hl.tuple([dups_ht._dup, dups_ht.rank])), key=lambda x: x[1]).map(lambda x: x[0])
    )
    dups_ht = dups_ht.group_by(dups_ht.dup_set).aggregate(
        **dups_cols
    )

    if isinstance(dups_ht.kept, hl.expr.StructExpression):
        dups_ht = dups_ht.key_by(**dups_ht.kept).drop('kept')
    else:
        dups_ht = dups_ht.key_by(s=dups_ht.kept)  # Since there is no defined name in the case of a non-struct type, use `s`
    return dups_ht


def compute_related_samples_to_drop(
        relatedness_ht: hl.Table,
        rank_ht: hl.Table,
        kin_threshold: float,
        filtered_samples: Optional[hl.expr.SetExpression] = None,
        min_related_hard_filter: Optional[int] = None
 ) -> hl.Table:
    """
    Computes a Table with the list of samples to drop (and their global rank) to get the maximal independent set of unrelated samples.

    Note:
    - `relatedness_ht` should be keyed by exactly two fields of the same type, identifying the pair of samples for each row.
    - `rank_ht` should be keyed by a single key of the same type as a single sample identifier in `relatedness_ht`.

    :param Table relatedness_ht: relatedness HT, as produced by e.g. pc-relate
    :param float kin_threshold: Kinship threshold to consider two samples as related
    :param Table rank_ht: Table with a global rank for each sample (smaller is preferred)
    :param SetExpression filtered_samples: An optional set of samples to exclude (e.g. these samples were hard-filtered)  These samples will then appear in the resulting samples to drop.
    :param int min_related_hard_filter: If provided, any sample that is related to more samples than this parameter will be filtered prior to computing the maximal independent set and appear in the results.
    :return: A Table with the list of the samples to drop along with their rank.
    :rtype: Table
    """

    # Make sure that the key types are valid
    assert(len(list(relatedness_ht.key))== 2)
    assert(relatedness_ht.key[0].dtype == relatedness_ht.key[1].dtype)
    assert (len(list(rank_ht.key)) == 1)
    assert (relatedness_ht.key[0].dtype == rank_ht.key[0].dtype)

    logger.info(f"Filtering related samples using a kin threshold of {kin_threshold}")
    relatedness_ht = relatedness_ht.filter(relatedness_ht.kin > kin_threshold)

    filtered_samples_rel = set()
    if min_related_hard_filter is not None:
        logger.info(f"Computing samples related to too many individuals (>{min_related_hard_filter}) for exclusion")
        gbi = relatedness_ht.annotate(s=list(relatedness_ht.key))
        gbi = gbi.explode(gbi.s)
        gbi = gbi.group_by(gbi.s).aggregate(n=hl.agg.count())
        filtered_samples_rel = gbi.aggregate(hl.agg.filter(gbi.n > min_related_hard_filter, hl.agg.collect_as_set(gbi.s)))
        logger.info(f"Found {len(filtered_samples_rel)} samples with too many 1st/2nd degree relatives. These samples will be excluded.")

    if filtered_samples is not None:
        filtered_samples_rel = filtered_samples_rel.union(
            relatedness_ht.aggregate(
                hl.agg.explode(
                    lambda s: hl.agg.collect_as_set(s),
                    hl.array(list(relatedness_ht.key)).filter(lambda s: filtered_samples.contains(s))
                )
            )
        )

    if len(filtered_samples_rel) > 0:
        filtered_samples_lit = hl.literal(filtered_samples_rel)
        relatedness_ht = relatedness_ht.filter(
            filtered_samples_lit.contains(relatedness_ht.key[0]) |
            filtered_samples_lit.contains(relatedness_ht.key[1]),
            keep=False
        )

    logger.info("Annotating related sample pairs with rank.")
    i, j = list(relatedness_ht.key)
    relatedness_ht = relatedness_ht.key_by(s=relatedness_ht[i])
    relatedness_ht = relatedness_ht.annotate(
        **{i: hl.struct(
            s=relatedness_ht.s,
            rank=rank_ht[relatedness_ht.key].rank
        )}
    )
    relatedness_ht = relatedness_ht.key_by(s=relatedness_ht[j])
    relatedness_ht = relatedness_ht.annotate(
        **{j: hl.struct(
            s=relatedness_ht.s,
            rank=rank_ht[relatedness_ht.key].rank
        )}
    )
    relatedness_ht = relatedness_ht.key_by(i, j)
    relatedness_ht = relatedness_ht.drop('s')
    relatedness_ht = relatedness_ht.persist()

    related_samples_to_drop_ht = hl.maximal_independent_set(
        relatedness_ht[i],
        relatedness_ht[j],
        keep=False,
        tie_breaker=lambda l,r: l.rank - r.rank
    )
    related_samples_to_drop_ht = related_samples_to_drop_ht.key_by()
    related_samples_to_drop_ht = related_samples_to_drop_ht.select(**related_samples_to_drop_ht.node)
    related_samples_to_drop_ht = related_samples_to_drop_ht.key_by('s')

    if len(filtered_samples_rel) > 0:
        related_samples_to_drop_ht = related_samples_to_drop_ht.union(
            hl.Table.parallelize(
                [hl.struct(s=s, rank=hl.null(hl.tint64)) for s in filtered_samples_rel],
                key='s'
            )
        )

    return related_samples_to_drop_ht


def run_pca_with_relateds(
        qc_mt: hl.MatrixTable,
        related_samples_to_drop: Optional[hl.Table],
        n_pcs: int = 10,
        autosomes_only: bool = True
) -> Tuple[List[float], hl.Table, hl.Table]:
    """
    First runs PCA excluding the given related samples,
    then projects these samples in the PC space to return scores for all samples.

    The `related_samples_to_drop` Table has to be keyed by the sample ID and all samples present in this
    table will be excluded from the PCA.

    The loadings Table returned also contains a `pca_af` annotation which is the allele frequency
    used for PCA. This is useful to project other samples in the PC space.

    :param MatrixTable qc_mt: Input QC MT
    :param Table related_samples_to_drop: Optional table of related samples to drop
    :param int n_pcs: Number of PCs to compute
    :param bool autosomes_only: Whether to run the analysis on autosomes only
    :return: eigenvalues, scores and loadings
    :rtype: (list of float, Table, Table)
    """

    unrelated_mt = qc_mt.persist()

    if autosomes_only:
        unrelated_mt = filter_to_autosomes(unrelated_mt)

    if related_samples_to_drop:
        unrelated_mt = qc_mt.filter_cols(hl.is_missing(related_samples_to_drop[qc_mt.col_key]))

    pca_evals, pca_scores, pca_loadings = hl.hwe_normalized_pca(unrelated_mt.GT, k=n_pcs, compute_loadings=True)
    pca_af_ht = unrelated_mt.annotate_rows(pca_af=hl.agg.mean(unrelated_mt.GT.n_alt_alleles()) / 2).rows()
    pca_loadings = pca_loadings.annotate(pca_af=pca_af_ht[pca_loadings.key].pca_af)  # TODO: Evaluate if needed to write results at this point if relateds or not

    if not related_samples_to_drop:
        return pca_evals, pca_scores, pca_loadings
    else:
        pca_loadings = pca_loadings.persist()
        pca_scores = pca_scores.persist()
        related_mt = qc_mt.filter_cols(hl.is_defined(related_samples_to_drop[qc_mt.col_key]))
        related_scores = pc_project(related_mt, pca_loadings)
        pca_scores = pca_scores.union(related_scores)
        return pca_evals, pca_scores, pca_loadings


def compute_stratified_metrics_filter(
        ht: hl.Table,
        qc_metrics: Dict[str, hl.expr.NumericExpression],
        strata: Optional[Dict[str, hl.expr.Expression]] = None,
        lower_threshold: float = 4.0,
        upper_threshold: float = 4.0,
        metric_threshold: Optional[Dict[str, Tuple[float, float]]] = None,
        filter_name: str = 'qc_metrics_filters'
) -> hl.Table:
    """
    Compute median, MAD, and upper and lower thresholds for each metric used in outlier filtering

    :param MatrixTable ht: HT containing relevant sample QC metric annotations
    :param dict of str -> qc_metrics: list of metrics (name and expr) for which to compute the critical values for filtering outliers
    :param list of str strata: List of annotations used for stratification. These metrics should be discrete types!
    :param float lower_threshold: Lower MAD threshold
    :param float upper_threshold: Upper MAD threshold
    :param dict str -> (float, float) metric_threshold: Can be used to specify different (lower, upper) thresholds for one or more metrics
    :param str filter_name: Name of resulting filters annotation
    :return: Table grouped by strata, with upper and lower threshold values computed for each sample QC metric
    :rtype: Table
    """

    _metric_threshold = {metric: (lower_threshold, upper_threshold) for metric in qc_metrics}
    if metric_threshold is not None:
        _metric_threshold.update(metric_threshold)

    def make_filters_expr(ht: hl.Table, qc_metrics: Iterable[str]) -> hl.expr.SetExpression:
        return hl.set(
            hl.filter(
                lambda x: hl.is_defined(x),
                [hl.or_missing(ht[f'fail_{metric}'], metric) for metric in qc_metrics]
            )
        )

    if strata is None:
        strata = {}

    ht = ht.select(**qc_metrics, **strata).key_by('s').persist()

    agg_expr = hl.struct(**{
        metric: hl.bind(
            lambda x: x.annotate(
                lower=x.median - _metric_threshold[metric][0] * x.mad,
                upper=x.median + _metric_threshold[metric][1] * x.mad
            ),
            get_median_and_mad_expr(ht[metric])
        ) for metric in qc_metrics
    })

    if strata:
        ht = ht.annotate_globals(qc_metrics_stats=ht.aggregate(hl.agg.group_by(hl.tuple([ht[x] for x in strata]), agg_expr), _localize=False))
        metrics_stats_expr = ht.qc_metrics_stats[hl.tuple([ht[x] for x in strata])]
    else:
        ht = ht.annotate_globals(qc_metrics_stats=ht.aggregate(agg_expr, _localize=False))
        metrics_stats_expr = ht.qc_metrics_stats

    fail_exprs = {
        f'fail_{metric}':
            (ht[metric] <= metrics_stats_expr[metric].lower) |
            (ht[metric] >= metrics_stats_expr[metric].upper)
        for metric in qc_metrics}
    ht = ht.transmute(**fail_exprs)
    stratified_filters = make_filters_expr(ht, qc_metrics)
    return ht.annotate(**{filter_name: stratified_filters})


def compute_qc_metrics_residuals(
        ht: hl.Table,
        pc_scores: hl.expr.ArrayNumericExpression,
        qc_metrics: Dict[str, hl.expr.NumericExpression],
        use_pc_square: bool = True,
        n_pcs: Optional[int] = None,
        regression_sample_inclusion_expr: hl.expr.BooleanExpression = hl.bool(True)
) -> hl.Table:
    """
    Computes QC metrics residuals after regressing out PCs (and optionally PC^2)

    Note: The `regression_sample_inclusion_expr` can be used to select a subset of the samples to include in the regression calculation.
    Residuals are always computed for all samples.

    :param Table ht: Input sample QC metrics HT
    :param ArrayNumericExpressoin pc_scores: The expression in the input HT that stores the PC scores
    :param dict of str -> NumericExpression qc_metrics: A dictionary with the name of each QC metric to compute residuals for and their corresponding expression in the input HT.
    :param bool use_pc_square: Whether to  use PC^2 in the regression or not
    :param int n_pcs: Numer of PCs to use. If not set, then all PCs in `pc_scores` are used.
    :param BooleanExpression regression_sample_inclusion_expr: An optional expression to select samples to include in the regression calculation.
    :return: Table with QC metrics residuals
    :rtype: Table
    """

    # Annotate QC HT with fields necessary for computation
    _sample_qc_ht = ht.select(
        **qc_metrics,
        scores=pc_scores,
        _keep=regression_sample_inclusion_expr
    )

    # If n_pcs wasn't provided, use all PCs
    if n_pcs is None:
        n_pcs = _sample_qc_ht.aggregate(hl.agg.min(hl.len(_sample_qc_ht.scores)))

    logger.info("Computing regressed QC metrics filters using {} PCs for metrics: {}".format(
        n_pcs,
        ', '.join(qc_metrics)
    ))

    # Prepare regression variables, adding 1.0 first for the intercept
    # Adds square of variables if use_pc_square is true
    x_expr = [1.0] + [_sample_qc_ht.scores[i] for i in range(0, n_pcs)]
    if use_pc_square:
        x_expr.extend([_sample_qc_ht.scores[i] * _sample_qc_ht.scores[i] for i in range(0, n_pcs)])

    # Compute linear regressions
    lms = _sample_qc_ht.aggregate(
        hl.struct(
            **{metric:
                hl.agg.filter(
                    _sample_qc_ht._keep,
                    hl.agg.linreg(
                    y=_sample_qc_ht[metric],
                    x=x_expr
                )
             ) for metric in qc_metrics
            }
        )
    )

    _sample_qc_ht = _sample_qc_ht.annotate_globals(
        lms=lms
    ).persist()

    # Compute residuals
    def get_lm_prediction_expr(metric: str):
        lm_pred_expr = _sample_qc_ht.lms[metric].beta[0] + hl.sum(
                hl.range(n_pcs).map(lambda i: _sample_qc_ht.lms[metric].beta[i + 1] * _sample_qc_ht.scores[i])
        )
        if use_pc_square:
            lm_pred_expr = lm_pred_expr + hl.sum(
                hl.range(n_pcs).map(lambda i: _sample_qc_ht.lms[metric].beta[i + n_pcs + 1] * _sample_qc_ht.scores[i] * _sample_qc_ht.scores[i])
        )
        return lm_pred_expr

    residuals_ht = _sample_qc_ht.select(
        **{
            f'{metric}_residual': _sample_qc_ht[metric] - get_lm_prediction_expr(metric)
            for metric in _sample_qc_ht.lms
        }
    )

    return residuals_ht.persist()


def flatten_duplicate_samples_ht(dups_ht: hl.Table) -> hl.Table:
    """
    Flattens the result of `filter_duplicate_samples`, so that each line contains a single sample.
    An additional annotation is added: `dup_filtered` indicating which of the duplicated samples was kept.

    Note that this assumes that the type of the table key is the same as the type of the `filtered` array.

    :param Table dups_ht: Input HT
    :return: Flattened HT
    :rtype: Table
    """
    dups_ht = dups_ht.annotate(
        dups=hl.array([(dups_ht.key, False)]).extend(
            dups_ht.filtered.map(lambda x: (x, True))
        )
    )
    dups_ht = dups_ht.explode('dups')
    dups_ht = dups_ht.key_by()
    return dups_ht.select(s=dups_ht.dups[0], dup_filtered=dups_ht.dups[1]).key_by('s')


def add_filters_expr(
        filters: Dict[str, hl.expr.BooleanExpression],
        current_filters: hl.expr.SetExpression = None
) -> hl.expr.SetExpression:
    """
    Creates an expression to create or add filters.
    For each entry in the `filters` dictionary, if the value evaluates to `True`,
    then the key is added as a filter name.

    Current filters are kept if provided using `current_filters`

    :param dict of str -> BooleanExpression filters: The filters and their expressions
    :param SetExpression current_filters: The set of current filters
    :return: An expression that can be used to annotate the filters
    :rtype: SetExpression
    """
    if current_filters is None:
        current_filters = hl.empty_set(hl.tstr)

    return hl.fold(
        lambda x, y: x.union(y),
        current_filters,
        [
            hl.cond(filter_condition, hl.set([filter_name]), hl.empty_set(hl.tstr))
            for filter_name, filter_condition in filters.items()
        ]
    )


def merge_sample_qc_expr(sample_qc_exprs: List[hl.expr.StructExpression]) -> hl.expr.StructExpression:
    """
    Creates an expression that merges results from non-overlapping strata of hail.sample_qc
    E.g.:
    - Compute autosomes and sex chromosomes metrics separately, then merge results
    - Compute bi-allelic and multi-allelic metrics separately, then merge results

    Note regarding the merging of ``dp_stats`` and ``gq_stats``:
    Because ``n`` is needed to aggregate ``stdev``, ``n_called`` is used for this purpose.
    This should work very well on a standard GATK VCF and it essentially assumes that:
    - samples that are called have `DP` and `GQ` fields
    - samples that are not called do not have `DP` and `GQ` fields
    Even if these assumptions are broken for some genotypes, it shouldn't matter too much.

    :param list of StructExpression sample_qc_exprs: List of sample QC struct expressions for each stratification
    :return: Combined sample QC results
    :rtype: StructExpression
    """

    # List of metrics that can be aggregated by summing
    additive_metrics = [
        'n_called', 'n_not_called', 'n_hom_ref', 'n_het', 'n_hom_var', 'n_snp',
        'n_insertion', 'n_deletion', 'n_singleton', 'n_transition', 'n_transversion', 'n_star'
    ]

    # List of metrics that are ratio of summed metrics (name, nominator, denominator)
    ratio_metrics = [
        ('call_rate', 'n_called', 'n_not_called'),
        ('r_ti_tv', 'n_transition', 'n_transversion'),
        ('r_het_hom_var', 'n_het', 'n_hom_var'),
        ('r_insertion_deletion', 'n_insertion', 'n_deletion')
    ]

    # List of metrics that are struct generated by a stats counter
    stats_metrics = ['gq_stats', 'dp_stats']

    # Gather metrics present in sample qc fields
    sample_qc_fields = set(sample_qc_exprs[0])
    for sample_qc_expr in sample_qc_exprs[1:]:
        sample_qc_fields = sample_qc_fields.union(set(sample_qc_expr))

    # Merge additive metrics in sample qc fields
    merged_exprs = {
        metric: hl.sum([sample_qc_expr[metric] for sample_qc_expr in sample_qc_exprs])
        for metric in additive_metrics if metric in sample_qc_fields
    }

    # Merge ratio metrics in sample qc fields
    merged_exprs.update({
        metric: hl.float64(divide_null(merged_exprs[nom], merged_exprs[denom]))
        for metric, nom, denom in ratio_metrics if nom in sample_qc_fields and denom in sample_qc_fields
    })

    # Merge stats counter metrics in sample qc fields
    # Use n_called as n for DP and GQ stats
    if 'n_called' in sample_qc_fields:
        merged_exprs.update({
            metric: merge_stats_counters_expr([
                sample_qc_expr[metric].annotate(n=sample_qc_expr.n_called)
                for sample_qc_expr in sample_qc_exprs
            ]).drop('n')
            for metric in stats_metrics
        })

    return hl.struct(**merged_exprs)


def compute_stratified_sample_qc(
        mt: hl.MatrixTable,
        strata: Dict[str, hl.expr.BooleanExpression],
        tmp_ht_prefix: Optional[str],
        gt_expr: Optional[hl.expr.CallExpression]
) -> hl.Table:
    """
    Runs hl.sample_qc on different strata and then also merge the results into a single expression.
    Note that strata should be non-overlapping,  e.g. SNV vs indels or bi-allelic vs multi-allelic

    :param MatrixTable mt: Input MT
    :param dict of str -> BooleanExpression strata: Strata names and filtering expressions
    :param str tmp_ht_prefix: Optional path prefix to write the intermediate strata results to (recommended for larger datasets)
    :param CallExpression gt_expr: Optional entry field storing the genotype (if not specified, then it is assumed that it is stored in mt.GT)
    :return: Sample QC table, including strat-specific numbers
    :rtype: Table
    """
    mt = mt.select_rows(
        **strata
    )

    if gt_expr is not None:
        mt = mt.select_entries(GT=gt_expr)
    else:
        mt = mt.select_entries('GT')

    strat_hts = {}
    for strat in strata:
        strat_sample_qc_ht = hl.sample_qc(mt.filter_rows(mt[strat])).cols()
        if tmp_ht_prefix is not None:
            strat_sample_qc_ht = strat_sample_qc_ht.checkpoint(tmp_ht_prefix + f'_{strat}.ht', overwrite=True)
        else:
            strat_sample_qc_ht = strat_sample_qc_ht.persist()
        strat_hts[strat] = strat_sample_qc_ht

    sample_qc_ht = strat_hts.pop(list(strata)[0])
    sample_qc_ht = sample_qc_ht.select(
        **{f'{list(strata)[0]}_sample_qc': sample_qc_ht.sample_qc},
        **{f'{strat}_sample_qc': strat_hts[strat][sample_qc_ht.key].sample_qc for strat in list(strata)[1:]}
    )
    sample_qc_ht = sample_qc_ht.annotate(
        sample_qc=merge_sample_qc_expr(list(sample_qc_ht.row_value.values()))
    )

    return sample_qc_ht


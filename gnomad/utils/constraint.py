# noqa: D100
# cSpell: disable
from typing import Any, Optional, Tuple, Union
import logging

import hail as hl

logging.basicConfig(
    format="%(asctime)s (%(name)s %(lineno)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger("constraint_pipeline")
logger.setLevel(logging.INFO)


def annotate_with_mu(
    ht: hl.Table,
    mutation_ht: hl.Table,
    output_col: str = "mu_snp",
) -> hl.Table:
    """
    Annotate SNP mutation rate for the input Table.

    :param ht: Input Table.
    :param mutation_ht: Mutation rate Table.
    :param output_col: Name for mutational rate annotation. Defaults to 'mu_snp'.
    :return: Table with mutational rate annotation added (default name for annotation is 'mu_snp').
    """
    keys = mutation_ht.key
    mu = hl.literal(
        mutation_ht.aggregate(
            hl.dict(
                hl.agg.collect(
                    (hl.struct(**{k: mutation_ht[k] for k in keys}), mutation_ht.mu_snp)
                )
            )
        )
    )
    mu = mu.get(hl.struct(**{k: ht[k] for k in keys}))
    return ht.annotate(
        **{output_col: hl.case().when(hl.is_defined(mu), mu).or_error("Missing mu")}
    )


def count_variants(
    ht: hl.Table,
    count_singletons: bool = False,
    count_downsamplings: Optional[Tuple[str]] = (),
    additional_grouping: Optional[Tuple[str]] = (),
    partition_hint: int = 100,
    omit_methylation: bool = False,
    use_table_group_by: bool = False,
    singleton_expr: hl.expr.BooleanExpression = None,
    max_af: Optional[float] = None,
    freq_expr: hl.expr.ArrayExpression = None,
    freq_meta_expr: hl.expr.ArrayExpression = None,
) -> Union[hl.Table, Any]:
    """
    Count number of observed or possible variants by context, ref, alt, and optionally methylation_level.

    :param ht: Input Hail Table.
    :param count_singletons: Whether to count singletons. Defaults to False.
    :param count_downsamplings: List of populations to use for downsampling counts. Defaults to ().
    :param additional_grouping: Additional features to group by. i.e. exome_coverage. Defaults to ().
    :param partition_hint: Target number of partitions for aggregation. Defaults to 100.
    :param omit_methylation: Whether to omit 'methylation_level' from the grouping when counting variants. Defaults to False.
    :param use_table_group_by: Whether to force grouping. Defaults to False.
    :param singleton_expression: Expression for defining a singleton. Defaults to None.
    :param max_af: Maximum variant AF to keep. By default, no AF cutoff is applied.
    :param freq_expr: ArrayExpression of Structs with with 'AC' and 'AF' annotations.
    :param freq_meta_expr: ArrayExpression of meta dictionaries corresponding to freq_expr.
    :return: Table including 'variant_count' and downsampling counts if requested.
    """
    if freq_expr is None and (
        count_downsamplings or max_af or (count_singletons and singleton_expr is None)
    ):
        logger.warning(
            "freq_expr was not provided, using 'freq' as the frequency annotation."
        )
        freq_expr = ht.freq
    if count_downsamplings and freq_meta_expr is None:
        logger.warning(
            "freq_meta_expr was not provided, using 'freq_meta' as the frequency metadata annotation."
        )
        freq_meta_expr = ht.freq_meta
    if count_singletons and singleton_expr is None:
        logger.warning(
            "count_singletons is True and singleton_expr was not provided, using freq_expr[0].AC == 1 as the singleton expression."
        )
        # Slower, but more flexible (allows for downsampling agg's)
        singleton_expr = freq_expr[0].AC == 1

    grouping = hl.struct(context=ht.context, ref=ht.ref, alt=ht.alt)
    if not omit_methylation:
        grouping = grouping.annotate(methylation_level=ht.methylation_level)
    for group in additional_grouping:
        grouping = grouping.annotate(**{group: ht[group]})

    agg = {"variant_count": hl.agg.count_where(freq_expr.AF <= max_af) if max_af else hl.agg.count()}
    if count_singletons:
        agg["singleton_count"] = hl.agg.count_where(singleton_expr)
        
    if count_downsamplings:       
        for pop in count_downsamplings:
            agg[f"downsampling_counts_{pop}"] = downsampling_counts_expr(freq_expr, freq_meta_expr, pop, max_af=max_af)
            if count_singletons:
                agg[f"singleton_downsampling_counts_{pop}"] = downsampling_counts_expr(freq_expr, freq_meta_expr, pop, singleton=True)
    
    if use_table_group_by:
        return ht.group_by(**grouping).partition_hint(partition_hint).aggregate(**agg)
    else:
        return ht.aggregate(hl.struct(**{field: hl.agg.group_by(grouping, agg[field]) for field in agg}))


def downsampling_counts_expr(
    freq_expr: hl.expr.ArrayExpression,
    freq_meta_expr: hl.expr.ArrayExpression,
    pop: str = "global",
    variant_quality: str = "adj",
    singleton: bool = False,
    max_af: Optional[float] = None,
) -> hl.expr.ArrayExpression:
    """
    Downsample the variant count per given population.

    :param freq_expr: ArrayExpression of Structs with with 'AC' and 'AF' annotations.
    :param freq_meta_expr: ArrayExpression of meta dictionaries corresponding to freq_expr.
    :param pop: Population. Defaults to 'global'.
    :param variant_quality: Variant quality for "group" key. Defaults to 'adj'.
    :param singleton: Whether to sum only alleles that are singletons. Defaults to False.
    :param max_af: Maximum variant allele frequency to keep. By default no allele frequency cutoff is applied.
    :return: Downsampling count for specified population.
    """
    indices = hl.enumerate(freq_meta_expr).filter(
        lambda f: (f[1].size() == 3)
        & (f[1].get("group") == variant_quality)
        & (f[1].get("pop") == pop)
        & f[1].contains("downsampling")
    )
    sorted_indices = hl.sorted(indices, key=lambda f: hl.int(f[1]["downsampling"])).map(
        lambda x: x[0]
    )

    def _get_criteria(i):
        if singleton:
            return hl.int(freq_expr[i].AC == 1)
        elif max_af:
            return hl.int((freq_expr[i].AC > 0) & (freq_expr[i].AF <= max_af))
        else:
            return hl.int(freq_expr[i].AC > 0)

    return hl.agg.array_sum(hl.map(_get_criteria, sorted_indices))


import re
import sys
import logging
import gzip
import os

import hail as hl
from hail.expr.expressions import *
from collections import defaultdict, namedtuple, OrderedDict
from pprint import pprint, pformat
import argparse
from typing import *
import json

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def filter_to_adj(mt: hl.MatrixTable) -> hl.MatrixTable:
    """
    Filter genotypes to adj criteria
    """
    if 'adj' not in list(mt.entry):
        mt = annotate_adj(mt)
    mt = mt.filter_entries(mt.adj)
    return mt.drop(mt.adj)


def annotate_adj(mt: hl.MatrixTable) -> hl.MatrixTable:
    """
    Annotate genotypes with adj criteria (assumes diploid)
    """
    adj_gq = 20
    adj_dp = 10
    adj_ab = 0.2

    return mt.annotate_entries(adj=(mt.GQ >= adj_gq) & (mt.DP >= adj_dp) & (
                                   ~mt.GT.is_het() |
                                   ((mt.GT[0] == 0) & (mt.AD[mt.GT[1]] / mt.DP >= adj_ab)) |
                                   ((mt.GT[0] > 0) & (mt.AD[mt.GT[0]] / mt.DP >= adj_ab) &
                                    (mt.AD[mt.GT[1]] / mt.DP >= adj_ab))
                               ))


def add_variant_type(alt_alleles: hl.expr.ArrayExpression) -> hl.expr.StructExpression:
    """
    Get Struct of variant_type and n_alt_alleles from ArrayExpression of Strings (all alleles)
    """
    ref = alt_alleles[0]
    alts = alt_alleles[1:]
    non_star_alleles = hl.filter(lambda a: a != '*', alts)
    return hl.struct(variant_type=hl.cond(
        hl.all(lambda a: hl.is_snp(ref, a), non_star_alleles),
        hl.cond(hl.len(non_star_alleles) > 1, "multi-snv", "snv"),
        hl.cond(
            hl.all(lambda a: hl.is_indel(ref, a), non_star_alleles),
            hl.cond(hl.len(non_star_alleles) > 1, "multi-indel", "indel"),
            "mixed")
    ), n_alt_alleles=hl.len(non_star_alleles))


def adjust_sex_ploidy(mt: hl.MatrixTable, sex_expr: hl.expr.StringExpression,
                      male_str: str = 'male', female_str: str = 'female') -> hl.MatrixTable:
    """
    Converts males to haploid on non-PAR X/Y, sets females to missing on Y

    :param MatrixTable mt: Input MatrixTable
    :param StringExpression sex_expr: Expression pointing to sex in MT (if not male_str or female_str, no change)
    :param str male_str: String for males (default 'male')
    :param str female_str: String for females (default 'female')
    :return: MatrixTable with fixed ploidy for sex chromosomes
    :rtype: MatrixTable
    """
    male = sex_expr == male_str
    female = sex_expr == female_str
    x_nonpar = mt.locus.in_x_nonpar()
    y_par = mt.locus.in_y_par()
    y_nonpar = mt.locus.in_y_nonpar()
    return mt.annotate_entries(
        GT=hl.case(missing_false=True)
        .when(female & (y_par | y_nonpar), hl.null(hl.tcall))
        .when(male & (x_nonpar | y_nonpar) & mt.GT.is_het(), hl.null(hl.tcall))
        .when(male & (x_nonpar | y_nonpar), hl.call(mt.GT[0], phased=False))
        .default(mt.GT)
    )


def add_popmax_expr(freq: hl.expr.ArrayExpression, populations: Set[str]) -> hl.expr.ArrayExpression:
    """
    Calculates popmax (add an additional entry into freq with popmax: pop)

    :param ArrayExpression freq: ArrayExpression of Structs with ['ac', 'an', 'hom', 'meta']
    :param set of str populations: Set of populations over which to calculate popmax
    :return: Frequency data with annotated popmax
    :rtype: ArrayExpression
    """
    pops_to_use = hl.literal(populations)
    freq_filtered = hl.filter(lambda f: (f.meta.size() == 2) & (f.meta.get('group') == 'adj') &
                                        pops_to_use.contains(f.meta.get('pop')), freq)
    sorted_freqs = hl.sorted(freq_filtered, key=lambda x: x.AC[1] / x.AN, reverse=True)
    return hl.cond(hl.len(sorted_freqs) > 0, freq.append(
        hl.struct(AC=sorted_freqs[0].AC, AF=sorted_freqs[0].AF, AN=sorted_freqs[0].AN,
                  homozygote_count=sorted_freqs[0].homozygote_count,
                  meta={'popmax': sorted_freqs[0].meta['pop']})), freq)


def get_projectmax(mt: hl.MatrixTable, loc: hl.expr.StringExpression) -> hl.MatrixTable:
    """
    First pass of projectmax (returns aggregated MT with project_max field)

    :param MatrixTable mt: Input MT
    :param StringExpression loc: Column expression location of project ID (e.g. mt.meta.pid)
    :return: Frequency data with annotated project_max
    :rtype: MatrixTable
    """
    mt = mt.annotate_cols(project=loc)
    agg_mt = mt.group_cols_by(mt.project).aggregate(callstats=hl.agg.call_stats(mt.GT, mt.alleles))
    return agg_mt.annotate_rows(project_max=hl.agg.take(hl.struct(**agg_mt.callstats, project=agg_mt.project),
                                                        5, -agg_mt.callstats.AF[1]))


def read_list_data(input_file: str) -> List[str]:
    if input_file.startswith('gs://'):
        hl.hadoop_copy(input_file, 'file:///' + input_file.split("/")[-1])
        f = gzip.open("/" + os.path.basename(input_file)) if input_file.endswith('gz') else open("/" + os.path.basename(input_file))
    else:
        f = gzip.open(input_file) if input_file.endswith('gz') else open(input_file)
    output = []
    for line in f:
        output.append(line.strip())
    f.close()
    return output


def filter_by_frequency(t: Union[hl.MatrixTable, hl.Table], direction: str,
                        frequency: float = None, allele_count: int = None,
                        population: str = None, subpop: str = None, downsampling: int = None,
                        keep: bool = True, adj: bool = True) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter MatrixTable or Table with gnomAD-format frequency data (assumed bi-allelic/split)
    (i.e. Array[Struct(Array[AC], Array[AF], AN, homozygote_count, meta)])
    At least one of frequency or allele_count is required.
    Subpop can be specified without a population if desired.

    :param MatrixTable or Table t: Input MatrixTable or Table
    :param str direction: One of "above", "below", and "equal" (how to apply the filter)
    :param float frequency: Frequency to filter by (one of frequency or allele_count is required)
    :param int allele_count: Allele count to filter by (one of frequency or allele_count is required)
    :param str population: Population in which to filter frequency
    :param str subpop: Sub-population in which to filter frequency
    :param int downsampling: Downsampling in which to filter frequency
    :param bool keep: Whether to keep rows passing this frequency (passed to filter_rows)
    :param bool adj: Whether to use adj frequency
    :return: Filtered MatrixTable or Table
    :rtype: MatrixTable or Table
    """
    if frequency is None and allele_count is None:
        raise ValueError('At least one of frequency or allele_count must be specified')
    if direction not in ('above', 'below', 'equal'):
        raise ValueError('direction needs to be one of "above", "below", or "equal"')
    group = 'adj' if adj else 'raw'
    criteria = [lambda f: f.meta.get('group') == group]
    if frequency is not None:
        if direction == 'above':
            criteria.append(lambda f: f.AF[1] > frequency)
        elif direction == 'below':
            criteria.append(lambda f: f.AF[1] < frequency)
        else:
            criteria.append(lambda f: f.AF[1] == frequency)
    if allele_count is not None:
        if direction == 'above':
            criteria.append(lambda f: f.AC[1] > allele_count)
        elif direction == 'below':
            criteria.append(lambda f: f.AC[1] < allele_count)
        else:
            criteria.append(lambda f: f.AC[1] == allele_count)
    size = 1
    if population:
        criteria.append(lambda f: f.meta.get('pop') == population)
        size += 1
    if subpop:
        criteria.append(lambda f: f.meta.get('subpop') == subpop)
        size += 1
        # If one supplies a subpop but not a population, this will ensure this gets it right
        if not population: size += 1
    if downsampling:
        criteria.append(lambda f: f.meta.get('downsampling') == str(downsampling))
        size += 1
        if not population:
            size += 1
            criteria.append(lambda f: f.meta.get('pop') == 'global')
        if subpop:
            raise Exception('No downsampling data for subpopulations implemented')
    criteria.append(lambda f: f.meta.size() == size)

    def combine_functions(func_list, x):
        cond = func_list[0](x)
        for c in func_list[1:]:
            cond &= c(x)
        return cond

    filt = lambda x: combine_functions(criteria, x)
    criteria = hl.any(filt, t.freq)
    return t.filter_rows(criteria, keep=keep) if isinstance(t, hl.MatrixTable) else t.filter(criteria, keep=keep)


def melt_kt(kt, columns_to_melt, key_column_name='variable', value_column_name='value'):
    """
    Go from wide to long, or from:

    +---------+---------+---------+
    | Variant | AC_NFE  | AC_AFR  |
    +=========+=========+=========+
    | 1:1:A:G |      1  |      8  |
    +---------+---------+---------+
    | 1:2:A:G |     10  |    100  |
    +---------+---------+---------+

    to:

    +---------+----------+--------+
    | Variant | variable | value  |
    +=========+==========+========+
    | 1:1:A:G |   AC_NFE |     1  |
    +---------+----------+--------+
    | 1:1:A:G |   AC_AFR |     8  |
    +---------+----------+--------+
    | 1:2:A:G |   AC_NFE |    10  |
    +---------+----------+--------+
    | 1:2:A:G |   AC_AFR |   100  |
    +---------+----------+--------+

    :param KeyTable kt: Input KeyTable
    :param list of str columns_to_melt: Which columns to spread out
    :param str key_column_name: What to call the key column
    :param str value_column_name: What to call the value column
    :return: melted Key Table
    :rtype: KeyTable
    return (kt
            .annotate('comb = [{}]'.format(', '.join(['{{k: "{0}", value: {0}}}'.format(x) for x in columns_to_melt])))
            .drop(columns_to_melt)
            .explode('comb')
            .annotate('{} = comb.k, {} = comb.value'.format(key_column_name, value_column_name))
            .drop('comb'))
    """
    raise NotImplementedError


def melt_kt_grouped(kt, columns_to_melt, value_column_names, key_column_name='variable'):
    """
    Go from wide to long for a group of variables, or from:

    +---------+---------+---------+---------+---------+
    | Variant | AC_NFE  | AC_AFR  | Hom_NFE | Hom_AFR |
    +=========+=========+=========+=========+=========+
    | 1:1:A:G |      1  |      8  |       0 |       0 |
    +---------+---------+---------+---------+---------+
    | 1:2:A:G |     10  |    100  |       1 |      10 |
    +---------+---------+---------+---------+---------+

    to:

    +---------+----------+--------+--------+
    | Variant |      pop |    AC  |   Hom  |
    +=========+==========+========+========+
    | 1:1:A:G |      NFE |     1  |     0  |
    +---------+----------+--------+--------+
    | 1:1:A:G |      AFR |     8  |     0  |
    +---------+----------+--------+--------+
    | 1:2:A:G |      NFE |    10  |     1  |
    +---------+----------+--------+--------+
    | 1:2:A:G |      AFR |   100  |    10  |
    +---------+----------+--------+--------+

    This is done with:

    columns_to_melt = {
        'NFE': ['AC_NFE', 'Hom_NFE'],
        'AFR': ['AC_AFR', 'Hom_AFR']
    }
    value_column_names = ['AC', 'Hom']
    key_column_name = 'pop'

    Note that len(value_column_names) == len(columns_to_melt[i]) for all in columns_to_melt

    :param KeyTable kt: Input KeyTable
    :param dict of list of str columns_to_melt: Which columns to spread out
    :param list of str value_column_names: What to call the value columns
    :param str key_column_name: What to call the key column
    :return: melted Key Table
    :rtype: KeyTable

    if any([len(value_column_names) != len(v) for v in columns_to_melt.values()]):
        logger.warning('Length of columns_to_melt sublist is not equal to length of value_column_names')
        logger.warning('value_column_names = %s', value_column_names)
        logger.warning('columns_to_melt = %s', columns_to_melt)

    # I think this goes something like this:
    fields = []
    for k, v in columns_to_melt.items():
        subfields = [': '.join(x) for x in zip(value_column_names, v)]
        field = '{{k: "{0}", {1}}}'.format(k, ', '.join(subfields))
        fields.append(field)

    split_text = ', '.join(['{0} = comb.{0}'.format(x) for x in value_column_names])

    return (kt
            .annotate('comb = [{}]'.format(', '.join(fields)))
            .drop([y for x in columns_to_melt.values() for y in x])
            .explode('comb')
            .annotate('{} = comb.k, {}'.format(key_column_name, split_text))
            .drop('comb'))
    """
    raise NotImplementedError


def get_rf_runs(data_type: str) -> Dict:
    """

    Loads RF run data from JSON file.

    :param str data_type: One of 'exomes' or 'genomes'
    :return: Dictionary containing the content of the JSON file, or an empty dictionary if the file wasn't found.
    :rtype: dict
    """
    
    from gnomad_hail.resources.variant_qc import rf_run_hash_path

    json_file = rf_run_hash_path(data_type)
    if hl.utils.hadoop_exists(json_file):
        with hl.hadoop_open(rf_run_hash_path(data_type)) as f:
            return json.load(f)
    else:
        logger.warning("File {json_file} could not be found. Returning empty RF run hash dict.")
        return {}


def pretty_print_runs(runs: Dict, label_col: str = 'rf_label', prediction_col_name: str = 'rf_prediction') -> None:
    """
    Prints the information for the RF runs loaded from the json file storing the RF run hashes -> info

    :param dict runs: Dictionary containing JSON input loaded from RF run file
    :param str label_col: Name of the RF label column
    :param str prediction_col_name: Name of the RF prediction column
    :return: Nothing -- only prints information
    :rtype: None
    """

    for run_hash, run_data in runs.items():
        print(f"\n=== {run_hash} ===")
        testing_results = run_data.pop('test_results') if 'test_results' in run_data else None
        # print(testing_results)
        print(json.dumps(run_data, sort_keys=True, indent=4, separators=(',', ': ')))
        if testing_results is not None:
            # Print results
            res_pd = pd.DataFrame(testing_results)
            res_pd = res_pd.pivot(index=label_col, columns=prediction_col_name, values='n')
            logger.info("Testing results:\n{}".format(pformat(res_pd)))

            
def add_full_rankings(ht: hl.Table, score_field: hl.expr.NumericExpression) -> hl.Table:
    """
    Add bi-allelic-only, singleton-only, and bi-allelic singleton-only variant QC rankings to a Hail Table
    containing variant annotations `was_split`, `info.AC`, and `a_index`

    :param Table ht: input Hail Table containing variants (with QC annotations) to be ranked
    :param NumericExpression score_field: the Table annotation by which ranking should be scored
    :return: Table with biallelic_rank, singleton_rank, and biallelic_singleton_rank added
    :rtype: Table
    """
    ht = ht.annotate(_score=score_field)

    # Rank all bi-allelics
    biallelic_ht = ht.filter(ht.was_split, keep=False)
    biallelic_ht = add_rank(biallelic_ht, biallelic_ht._score)

    # Rank all singletons
    singleton_ht = ht.filter(ht.info.AC[ht.a_index - 1] == 1)
    singleton_ht = add_rank(singleton_ht, singleton_ht._score)

    # Rank all bi-allelic singletons
    biallelic_singleton_ht = ht.filter((ht.info.AC[ht.a_index-1] == 1) & ~ht.was_split)  # NOTE: we are filtering to singletons across the entire callset, not just in high-quality samples
    biallelic_singleton_ht = add_rank(biallelic_singleton_ht, biallelic_singleton_ht._score)

    # Annotate and print sanity-check counts
    ht = ht.annotate(biallelic_rank=biallelic_ht[ht.key].rank, singleton_rank=singleton_ht[ht.key].rank, biallelic_singleton_rank=biallelic_singleton_ht[ht.key].rank)
    print(ht.aggregate(hl.struct(was_split=hl.agg.counter(ht.was_split),
                                 has_biallelic_rank=hl.agg.counter(hl.is_defined(ht.biallelic_rank)),
                                 was_singleton=hl.agg.counter(ht.info.AC[ht.a_index - 1] == 1),
                                 has_singleton_rank=hl.agg.counter(hl.is_defined(ht.singleton_rank)),
                                 was_split_singleton=hl.agg.counter((ht.info.AC[ht.a_index-1] == 1) & ~ht.was_split),
                                 has_biallelic_singleton_rank=hl.agg.counter(hl.is_defined(ht.biallelic_singleton_rank)))))
    return ht

def add_rank(ht: hl.Table, score_field: hl.expr.NumericExpression) -> hl.Table:
    """
    Adds an `rf_rank` row annotation based on its RF probability score.
    SNVs and Indels are ranked separately (both starting at 0)
    :param Table ht: Input RF results Hail Table
    :param NumericExpression score_field: the Table annotation by which ranking should be scored
    :return: Annotated Table
    :rtype: Table
    """
    ht = ht.annotate(_score=score_field).persist()
    rank_ht = ht.select(is_indel=hl.is_indel(ht.alleles[0], ht.alleles[1]), score=ht._score)
    n_snvs = rank_ht.aggregate(hl.agg.count_where(~rank_ht.is_indel))
    rank_ht = rank_ht.order_by(rank_ht.is_indel, hl.desc(rank_ht.score))
    rank_ht = rank_ht.add_index()
    rank_ht = rank_ht.annotate(idx=hl.cond(rank_ht.is_indel, rank_ht.idx - n_snvs, rank_ht.idx))
    return ht.annotate(rank=rank_ht[ht.key].idx).drop('_score')

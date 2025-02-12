# noqa: D100

import functools
import json
import logging
import operator
import os
import subprocess
from typing import Callable, List, Optional, Union

import hail as hl

from gnomad.resources.resource_utils import VersionedTableResource

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VEP_VERSIONS = ["101", "105"]
CURRENT_VEP_VERSION = VEP_VERSIONS[-1]
"""
Versions of VEP used in gnomAD data, the latest version is 105.
"""

# Note that these terms are current as of v105 with some included for backwards
# compatibility (VEP <= 75). The impact groupings are loosely based on VEP's categories
# but have been adjusted to better serve gnomAD's use cases.
# See: https://www.ensembl.org/info/genome/variation/prediction/predicted_data.html
CSQ_CODING_HIGH_IMPACT = [
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
]

CSQ_CODING_MEDIUM_IMPACT = [
    "start_lost",  # considered high impact in v105, previously medium
    "initiator_codon_variant",  # deprecated
    "transcript_amplification",  # considered high impact in v105, previously medium
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",  # new in v79
]

CSQ_CODING_LOW_IMPACT = [
    "splice_donor_5th_base_variant",  # new in v105
    "splice_region_variant",  # Considered low impact in v105, previously medium
    "splice_donor_region_variant",  # new in v105
    "splice_polypyrimidine_tract_variant",  # new in v105
    "incomplete_terminal_codon_variant",
    "start_retained_variant",  # new in v92
    "stop_retained_variant",
    "synonymous_variant",
    "coding_sequence_variant",
    "coding_transcript_variant",
]

CSQ_NON_CODING = [
    "mature_miRNA_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "non_coding_exon_variant",  # deprecated
    "intron_variant",
    "NMD_transcript_variant",
    "non_coding_transcript_variant",
    "nc_transcript_variant",  # deprecated
    "upstream_gene_variant",
    "downstream_gene_variant",
    "TFBS_ablation",
    "TFBS_amplification",
    "TF_binding_site_variant",
    "regulatory_region_ablation",
    "regulatory_region_amplification",
    "feature_elongation",
    "regulatory_region_variant",
    "feature_truncation",
    "intergenic_variant",
    "sequence_variant",
]

CSQ_ORDER = (
    CSQ_CODING_HIGH_IMPACT
    + CSQ_CODING_MEDIUM_IMPACT
    + CSQ_CODING_LOW_IMPACT
    + CSQ_NON_CODING
)

CSQ_CODING = CSQ_CODING_HIGH_IMPACT + CSQ_CODING_MEDIUM_IMPACT + CSQ_CODING_LOW_IMPACT
"""
Constant containing all coding consequences.
"""

CSQ_SPLICE = [
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_region_variant",
]
"""
Constant containing all splice consequences.
"""

POSSIBLE_REFS = ("GRCh37", "GRCh38")
"""
Constant containing supported references
"""

VEP_CONFIG_PATH = "file:///vep_data/vep-gcloud.json"
"""
Constant that contains the local path to the VEP config file
"""

VEP_CSQ_FIELDS = {
    "101": "Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|ALLELE_NUM|DISTANCE|STRAND|VARIANT_CLASS|MINIMISED|SYMBOL_SOURCE|HGNC_ID|CANONICAL|TSL|APPRIS|CCDS|ENSP|SWISSPROT|TREMBL|UNIPARC|GENE_PHENO|SIFT|PolyPhen|DOMAINS|HGVS_OFFSET|MOTIF_NAME|MOTIF_POS|HIGH_INF_POS|MOTIF_SCORE_CHANGE|LoF|LoF_filter|LoF_flags|LoF_info",
    "105": "Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|ALLELE_NUM|DISTANCE|STRAND|FLAGS|VARIANT_CLASS|SYMBOL_SOURCE|HGNC_ID|CANONICAL|MANE_SELECT|MANE_PLUS_CLINICAL|TSL|APPRIS|CCDS|ENSP|UNIPROT_ISOFORM|SOURCE|SIFT|PolyPhen|DOMAINS|miRNA|HGVS_OFFSET|PUBMED|MOTIF_NAME|MOTIF_POS|HIGH_INF_POS|MOTIF_SCORE_CHANGE|TRANSCRIPTION_FACTORS|LoF|LoF_filter|LoF_flags|LoF_info",
}
"""
Constant that defines the order of VEP annotations used in VCF export, currently stored in a dictionary with the VEP version as the key.
"""

VEP_CSQ_HEADER = (
    "Consequence annotations from Ensembl VEP. Format:"
    f" {VEP_CSQ_FIELDS[CURRENT_VEP_VERSION]}"
)
"""
Constant that contains description for VEP used in VCF export.
"""

LOFTEE_LABELS = ["HC", "LC", "OS"]
"""
Constant that contains annotations added by LOFTEE.
"""

LOF_CSQ_SET = {
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
}
"""
Set containing loss-of-function consequence strings.
"""


def get_vep_help(vep_config_path: Optional[str] = None):
    """
    Return the output of vep --help which includes the VEP version.

    .. warning::
        If no `vep_config_path` is supplied, this function will only work for Dataproc clusters
        created with `hailctl dataproc start --vep`. It assumes that the command is `/path/to/vep`.

    :param vep_config_path: Optional path to use as the VEP config file. If None, `VEP_CONFIG_URI` environment variable is used
    :return: VEP help string
    """
    if vep_config_path is None:
        vep_config_path = os.environ["VEP_CONFIG_URI"]

    with hl.hadoop_open(vep_config_path) as vep_config_file:
        vep_config = json.load(vep_config_file)
        vep_command = vep_config["command"]
        vep_help = subprocess.check_output([vep_command[0]]).decode("utf-8")
        return vep_help


def get_vep_context(ref: Optional[str] = None) -> VersionedTableResource:
    """
    Get VEP context resource for the genome build `ref`.

    :param ref: Genome build. If None, `hl.default_reference` is used
    :return: VEPed context resource
    """
    import gnomad.resources.grch37.reference_data as grch37
    import gnomad.resources.grch38.reference_data as grch38

    if ref is None:
        ref = hl.default_reference().name

    if ref not in POSSIBLE_REFS:
        raise ValueError(
            f'get_vep_context passed {ref}. Expected one of {", ".join(POSSIBLE_REFS)}'
        )

    vep_context = grch37.vep_context if ref == "GRCh37" else grch38.vep_context
    return vep_context


def vep_or_lookup_vep(
    ht, reference_vep_ht=None, reference=None, vep_config_path=None, vep_version=None
):
    """
    VEP a table, or lookup variants in a reference database.

    .. warning::
        If `reference_vep_ht` is supplied, no check is performed to confirm `reference_vep_ht` was
        generated with the same version of VEP / VEP configuration as the VEP referenced in `vep_config_path`.

    :param ht: Input Table
    :param reference_vep_ht: A reference database with VEP annotations (must be in top-level `vep`)
    :param reference: If reference_vep_ht is not specified, find a suitable one in reference (if None, grabs from hl.default_reference)
    :param vep_config_path: vep_config to pass to hl.vep (if None, a suitable one for `reference` is chosen)
    :param vep_version: Version of VEPed context Table to use (if None, the default `vep_context` resource will be used)
    :return: VEPed Table
    """
    if reference is None:
        reference = hl.default_reference().name

    if vep_config_path is None:
        vep_config_path = VEP_CONFIG_PATH

    vep_help = get_vep_help(vep_config_path)

    with hl.hadoop_open(vep_config_path) as vep_config_file:
        vep_config = vep_config_file.read()

    if reference_vep_ht is None:
        if reference not in POSSIBLE_REFS:
            raise ValueError(
                f"vep_or_lookup_vep got {reference}. Expected one of"
                f" {', '.join(POSSIBLE_REFS)}"
            )

        vep_context = get_vep_context(reference)
        if vep_version is None:
            vep_version = vep_context.default_version

        if vep_version not in vep_context.versions:
            logger.warning(
                "No VEPed context Table available for genome build %s and VEP"
                " version %s, all variants will be VEPed using the following"
                " VEP:\n%s",
                reference,
                vep_version,
                vep_help,
            )
            return hl.vep(ht, vep_config_path)

        logger.info(
            "Using VEPed context Table from genome build %s and VEP version %s",
            reference,
            vep_version,
        )

        reference_vep_ht = vep_context.versions[vep_version].ht()
        vep_context_help = hl.eval(reference_vep_ht.vep_help)
        vep_context_config = hl.eval(reference_vep_ht.vep_config)

        assert vep_help == vep_context_help, (
            "The VEP context HT version does not match the version referenced in the"
            f" VEP config file.\nVEP context:\n{vep_context_help}\n\n VEP"
            f" config:\n{vep_help}"
        )

        assert vep_config == vep_context_config, (
            "The VEP context HT configuration does not match the configuration in"
            f" {vep_config_path}.\nVEP context:\n{vep_context_config}\n\n Current"
            f" config:\n{vep_config}"
        )

    ht = ht.annotate(vep=reference_vep_ht[ht.key].vep)

    vep_ht = ht.filter(hl.is_defined(ht.vep))
    revep_ht = ht.filter(hl.is_missing(ht.vep))
    revep_ht = hl.vep(revep_ht, vep_config_path)
    if "vep_proc_id" in list(revep_ht.row):
        revep_ht = revep_ht.drop("vep_proc_id")
    if "vep_proc_id" in list(vep_ht.row):
        vep_ht = vep_ht.drop("vep_proc_id")

    vep_ht = vep_ht.annotate_globals(
        vep_version=f"v{vep_version}", vep_help=vep_help, vep_config=vep_config
    )

    return vep_ht.union(revep_ht)


def get_most_severe_consequence_expr(
    csq_expr: hl.expr.ArrayExpression,
    csq_order: Optional[List[str]] = None,
) -> hl.expr.StringExpression:
    """
    Get the most severe consequence from a collection of consequences.

    This is for a given transcript, as there are often multiple annotations for a single
    transcript: e.g. splice_region_variant&intron_variant -> splice_region_variant

    :param csq_expr: ArrayExpression of consequences.
    :param csq_order: Optional list indicating the order of VEP consequences, sorted
        from high to low impact. Default is None, which uses the value of the
        `CSQ_ORDER` global.
    :return: Most severe consequence in `csq_expr`.
    """
    if csq_order is None:
        csq_order = CSQ_ORDER
    csqs = hl.literal(csq_order)

    return csqs.find(lambda c: csq_expr.contains(c))


def add_most_severe_consequence_to_consequence(
    tc: Union[hl.expr.StructExpression, hl.expr.ArrayExpression],
    csq_order: Optional[List[str]] = None,
    most_severe_csq_field: str = "most_severe_consequence",
) -> Union[hl.expr.StructExpression, hl.expr.ArrayExpression]:
    """
    Add a `most_severe_consequence` field to a transcript consequence or array of transcript consequences.

    For a single transcript consequence, `tc` should be a StructExpression with a
    `consequence_terms` field, e.g. Struct(consequence_terms=['missense_variant']).
    For an array of transcript consequences, `tc` should be an ArrayExpression of
    StructExpressions with a `consequence_terms` field.

    :param tc: Transcript consequence or array of transcript consequences to annotate.
    :param csq_order: Optional list indicating the order of VEP consequences, sorted
        from high to low impact. Default is None, which uses the value of the
        `CSQ_ORDER` global.
    :param most_severe_csq_field: Field name to use for most severe consequence. Default
        is 'most_severe_consequence'.
    :return: Transcript consequence or array of transcript consequences annotated with
        the most severe consequence.
    """
    csq = lambda x: get_most_severe_consequence_expr(x.consequence_terms, csq_order)
    if isinstance(tc, hl.expr.StructExpression):
        return tc.annotate(**{most_severe_csq_field: csq(tc)})
    else:
        return tc.map(lambda x: x.annotate(**{most_severe_csq_field: csq(x)}))


def process_consequences(
    t: Union[hl.MatrixTable, hl.Table],
    vep_root: str = "vep",
    penalize_flags: bool = True,
    csq_order: Optional[List[str]] = None,
    has_polyphen: bool = True,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Add most_severe_consequence into [vep_root].transcript_consequences, and worst_csq_by_gene, any_lof into [vep_root].

    `most_severe_consequence` is the worst consequence for a transcript.

    Each transcript consequence is annotated with a `csq_score` which is a combination
    of the index of the consequence's `most_severe_consequence` in `csq_order` and an
    extra deduction for loss-of-function consequences, and polyphen predictions if
    `has_polyphen` is True. Lower scores translate to higher severity.

    The score adjustment is as follows:
        - lof == 'HC' & NO lof_flags (-1000 if penalize_flags, -500 if not)
        - lof == 'HC' & lof_flags (-500)
        - lof == 'OS' (-20)
        - lof == 'LC' (-10)
        - everything else (0)

    .. note::

        From gnomAD v4.0 on, the PolyPhen annotation was removed from the VEP Struct
        in the release HTs. When using this function with gnomAD v4.0 or later,
        set `has_polyphen` to False.

    :param t: Input Table or MatrixTable.
    :param vep_root: Root for VEP annotation (probably "vep").
    :param penalize_flags: Whether to penalize LOFTEE flagged variants, or treat them
        as equal to HC.
    :param csq_order: Optional list indicating the order of VEP consequences, sorted
        from high to low impact. Default is None, which uses the value of the
        `CSQ_ORDER` global.
    :param has_polyphen: Whether the input VEP Struct has a PolyPhen annotation which
        will be used to modify the consequence score. Default is True.
    :return: MT with better formatted consequences.
    """
    if csq_order is None:
        csq_order = CSQ_ORDER
    csqs = hl.literal(csq_order)

    # Assign a score to each consequence based on the order in csq_order.
    csq_dict = hl.literal(dict(zip(csq_order, range(len(csq_order)))))

    def _find_worst_transcript_consequence(
        tcl: hl.expr.ArrayExpression,
    ) -> hl.expr.StructExpression:
        """
        Find the worst transcript consequence in an array of transcript consequences.

        :param tcl: Array of transcript consequences.
        :return: Worst transcript consequence.
        """
        flag = 500
        no_flag = flag * (1 + penalize_flags)

        # Score each consequence based on the order in csq_order.
        score_expr = tcl.map(
            lambda tc: csq_dict[csqs.find(lambda x: x == tc.most_severe_consequence)]
        )

        # Determine the score adjustment based on the consequence's LOF and LOF flags.
        sub_expr = tcl.map(
            lambda tc: (
                hl.case(missing_false=True)
                .when((tc.lof == "HC") & hl.or_else(tc.lof_flags == "", True), no_flag)
                .when((tc.lof == "HC") & (tc.lof_flags != ""), flag)
                .when(tc.lof == "OS", 20)
                .when(tc.lof == "LC", 10)
                .default(0)
            )
        )

        # If requested, determine the score adjustment based on the consequence's
        # PolyPhen prediction.
        if has_polyphen:
            polyphen_sub_expr = tcl.map(
                lambda tc: (
                    hl.case(missing_false=True)
                    .when(tc.polyphen_prediction == "probably_damaging", 0.5)
                    .when(tc.polyphen_prediction == "possibly_damaging", 0.25)
                    .when(tc.polyphen_prediction == "benign", 0.1)
                    .default(0)
                )
            )
            sub_expr = hl.map(lambda s, ps: s + ps, sub_expr, polyphen_sub_expr)

        # Calculate the final consequence score.
        tcl = hl.map(
            lambda tc, s, ss: tc.annotate(csq_score=s - ss), tcl, score_expr, sub_expr
        )

        # Return the worst consequence based on the calculated score.
        return hl.or_missing(hl.len(tcl) > 0, hl.sorted(tcl, lambda x: x.csq_score)[0])

    # Annotate each transcript consequence with the 'most_severe_consequence'.
    transcript_csqs = add_most_severe_consequence_to_consequence(
        t[vep_root].transcript_consequences, csq_order
    )

    # Group transcript consequences by gene and find the worst consequence for each.
    gene_dict = transcript_csqs.group_by(lambda tc: tc.gene_symbol)
    worst_csq_gene = gene_dict.map_values(_find_worst_transcript_consequence).values()
    sorted_scores = hl.sorted(worst_csq_gene, key=lambda tc: tc.csq_score)

    # Filter transcript consequences to only include canonical transcripts and find the
    # worst consequence for each gene.
    canonical = transcript_csqs.filter(lambda csq: csq.canonical == 1)
    gene_canonical_dict = canonical.group_by(lambda tc: tc.gene_symbol)
    worst_csq_gene_canonical = gene_canonical_dict.map_values(
        _find_worst_transcript_consequence
    ).values()
    sorted_canonical_scores = hl.sorted(
        worst_csq_gene_canonical, key=lambda tc: tc.csq_score
    )

    # Annotate the HT/MT with the worst consequence for each gene and variant.
    vep_data = t[vep_root].annotate(
        transcript_consequences=transcript_csqs,
        worst_consequence_term=csqs.find(
            lambda c: transcript_csqs.map(
                lambda csq: csq.most_severe_consequence
            ).contains(c)
        ),
        worst_csq_by_gene=sorted_scores,
        worst_csq_for_variant=hl.or_missing(
            hl.len(sorted_scores) > 0, sorted_scores[0]
        ),
        worst_csq_by_gene_canonical=sorted_canonical_scores,
        worst_csq_for_variant_canonical=hl.or_missing(
            hl.len(sorted_canonical_scores) > 0, sorted_canonical_scores[0]
        ),
    )

    return (
        t.annotate_rows(**{vep_root: vep_data})
        if isinstance(t, hl.MatrixTable)
        else t.annotate(**{vep_root: vep_data})
    )


def filter_vep_to_canonical_transcripts(
    mt: Union[hl.MatrixTable, hl.Table],
    vep_root: str = "vep",
    filter_empty_csq: bool = False,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter VEP transcript consequences to those in the canonical transcript.

    :param mt: Input Table or MatrixTable.
    :param vep_root: Name used for VEP annotation. Default is 'vep'.
    :param filter_empty_csq: Whether to filter out rows where 'transcript_consequences' is empty. Default is False.
    :return: Table or MatrixTable with VEP transcript consequences filtered.
    """
    return filter_vep_transcript_csqs(
        mt, vep_root, synonymous=False, filter_empty_csq=filter_empty_csq
    )


def filter_vep_to_mane_select_transcripts(
    mt: Union[hl.MatrixTable, hl.Table],
    vep_root: str = "vep",
    filter_empty_csq: bool = False,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter VEP transcript consequences to those in the MANE Select transcript.

    :param mt: Input Table or MatrixTable.
    :param vep_root: Name used for VEP annotation. Default is 'vep'.
    :param filter_empty_csq: Whether to filter out rows where 'transcript_consequences' is empty. Default is False.
    :return: Table or MatrixTable with VEP transcript consequences filtered.
    """
    return filter_vep_transcript_csqs(
        mt,
        vep_root,
        synonymous=False,
        canonical=False,
        mane_select=True,
        filter_empty_csq=filter_empty_csq,
    )


def filter_vep_to_synonymous_variants(
    mt: Union[hl.MatrixTable, hl.Table],
    vep_root: str = "vep",
    filter_empty_csq: bool = False,
) -> Union[hl.MatrixTable, hl.Table]:
    """
    Filter VEP transcript consequences to those with a most severe consequence of 'synonymous_variant'.

    :param mt: Input Table or MatrixTable.
    :param vep_root: Name used for VEP annotation. Default is 'vep'.
    :param filter_empty_csq: Whether to filter out rows where 'transcript_consequences' is empty. Default is False.
    :return: Table or MatrixTable with VEP transcript consequences filtered.
    """
    return filter_vep_transcript_csqs(
        mt, vep_root, canonical=False, filter_empty_csq=filter_empty_csq
    )


def filter_vep_to_gene_list(
    t: Union[hl.MatrixTable, hl.Table],
    genes: List[str],
    match_by_gene_symbol: bool = False,
    vep_root: str = "vep",
    filter_empty_csq: bool = False,
):
    """
    Filter VEP transcript consequences to those in a set of genes.

    .. note::

       Filtering to a list of genes by their 'gene_id' or 'gene_symbol' will filter to
       all variants that are annotated to the gene, including
       ['upstream_gene_variant', 'downstream_gene_variant'], which will not be the
       same as if you filter to a gene interval. If you only want variants inside
       certain gene boundaries and a faster filter, you can first filter `t` to an
       interval list and then apply this filter.

    :param t: Input Table or MatrixTable.
    :param genes: Genes of interest to filter VEP transcript consequences to.
    :param match_by_gene_symbol: Whether to match values in `genes` to VEP transcript
        consequences by 'gene_symbol' instead of 'gene_id'. Default is False.
    :param vep_root: Name used for VEP annotation. Default is 'vep'.
    :param filter_empty_csq: Whether to filter out rows where 'transcript_consequences'
        is empty. Default is False.
    :return: Table or MatrixTable with VEP transcript consequences filtered.
    """
    return filter_vep_transcript_csqs(
        t,
        vep_root,
        synonymous=False,
        canonical=False,
        filter_empty_csq=filter_empty_csq,
        genes=genes,
        match_by_gene_symbol=match_by_gene_symbol,
    )


def vep_struct_to_csq(
    vep_expr: hl.expr.StructExpression,
    csq_fields: str = VEP_CSQ_FIELDS[CURRENT_VEP_VERSION],
    has_polyphen_sift: bool = True,
) -> hl.expr.ArrayExpression:
    """
    Given a VEP Struct, returns and array of VEP VCF CSQ strings (one per consequence in the struct).

    The fields and their order will correspond to those passed in `csq_fields`, which corresponds to the
    VCF header that is required to interpret the VCF CSQ INFO field.

    Note that the order is flexible and that all fields that are in the default value are supported.
    These fields will be formatted in the same way that their VEP CSQ counterparts are.

    While other fields can be added if their name are the same as those in the struct. Their value will be the result of calling
    hl.str(), so it may differ from their usual VEP CSQ representation.

    :param vep_expr: The input VEP Struct
    :param csq_fields: The | delimited list of fields to include in the CSQ (in that order), default is the CSQ fields of the CURRENT_VEP_VERSION.
    :param has_polyphen_sift: Whether the input VEP Struct has PolyPhen and SIFT annotations. Default is True.
    :return: The corresponding CSQ strings
    """
    _csq_fields = [f.lower() for f in csq_fields.split("|")]

    def get_csq_from_struct(
        element: hl.expr.StructExpression, feature_type: str
    ) -> hl.expr.StringExpression:
        # Most fields are 1-1, just lowercase
        fields = dict(element)

        # Add general exceptions
        fields.update(
            {
                "allele": element.variant_allele,
                "consequence": hl.delimit(element.consequence_terms, delimiter="&"),
                "feature_type": feature_type,
                "feature": (
                    element.transcript_id
                    if "transcript_id" in element
                    else (
                        element.regulatory_feature_id
                        if "regulatory_feature_id" in element
                        else (
                            element.motif_feature_id
                            if "motif_feature_id" in element
                            else ""
                        )
                    )
                ),
                "variant_class": vep_expr.variant_class,
            }
        )

        # Add exception for transcripts
        if feature_type == "Transcript":
            transcript_dict = {
                "canonical": hl.if_else(element.canonical == 1, "YES", ""),
                "ensp": element.protein_id,
                "gene": element.gene_id,
                "symbol": element.gene_symbol,
                "symbol_source": element.gene_symbol_source,
                "cdna_position": hl.str(element.cdna_start)
                + hl.if_else(
                    element.cdna_start == element.cdna_end,
                    "",
                    "-" + hl.str(element.cdna_end),
                ),
                "cds_position": hl.str(element.cds_start)
                + hl.if_else(
                    element.cds_start == element.cds_end,
                    "",
                    "-" + hl.str(element.cds_end),
                ),
                "mirna": hl.delimit(element.mirna, "&") if "mirna" in element else None,
                "protein_position": hl.str(element.protein_start)
                + hl.if_else(
                    element.protein_start == element.protein_end,
                    "",
                    "-" + hl.str(element.protein_end),
                ),
                "uniprot_isoform": (
                    hl.delimit(element.uniprot_isoform, "&")
                    if "uniprot_isoform" in element
                    else None
                ),
            }
            # Retain transcript dict updates only for fields that exist in the csq
            # fields.
            transcript_dict = {
                k: v
                for k, v in transcript_dict.items()
                if k in [x.lower() for x in csq_fields.split("|")]
            }
            fields.update(transcript_dict)

            if has_polyphen_sift:
                fields.update(
                    {
                        "sift": (
                            element.sift_prediction
                            + "("
                            + hl.format("%.3f", element.sift_score)
                            + ")"
                        ),
                        "polyphen": (
                            element.polyphen_prediction
                            + "("
                            + hl.format("%.3f", element.polyphen_score)
                            + ")"
                        ),
                    }
                )
            fields.update(
                {
                    "domains": hl.delimit(
                        element.domains.map(lambda d: d.db + ":" + d.name), "&"
                    ),
                }
            )
        elif feature_type == "MotifFeature":
            fields["motif_score_change"] = hl.format("%.3f", element.motif_score_change)
            if "transcription_factors" in element:
                fields["transcription_factors"] = hl.delimit(
                    element.transcription_factors, "&"
                )

        return hl.delimit(
            [hl.or_else(hl.str(fields.get(f, "")), "") for f in _csq_fields], "|"
        )

    csq = hl.empty_array(hl.tstr)
    for feature_field, feature_type in [
        ("transcript_consequences", "Transcript"),
        ("regulatory_feature_consequences", "RegulatoryFeature"),
        ("motif_feature_consequences", "MotifFeature"),
        ("intergenic_consequences", "Intergenic"),
    ]:
        csq = csq.extend(
            hl.or_else(
                vep_expr[feature_field].map(
                    lambda x: get_csq_from_struct(x, feature_type=feature_type)
                ),
                hl.empty_array(hl.tstr),
            )
        )

    return hl.or_missing(hl.len(csq) > 0, csq)


def get_most_severe_consequence_for_summary(
    ht: hl.Table,
    csq_order: List[str] = CSQ_ORDER,
    loftee_labels: List[str] = LOFTEE_LABELS,
) -> hl.Table:
    """
    Prepare a hail Table for summary statistics generation.

    Adds the following annotations:
        - most_severe_csq: Most severe consequence for variant
        - protein_coding: Whether the variant is present on a protein-coding transcript
        - lof: Whether the variant is a loss-of-function variant
        - no_lof_flags: Whether the variant has any LOFTEE flags (True if no flags)

    Assumes input Table is annotated with VEP and that VEP annotations have been filtered to canonical transcripts.

    :param ht: Input Table.
    :param csq_order: Order of VEP consequences, sorted from high to low impact. Default is CSQ_ORDER.
    :param loftee_labels: Annotations added by LOFTEE. Default is LOFTEE_LABELS.
    :return: Table annotated with VEP summary annotations.
    """

    def _get_most_severe_csq(
        csq_list: hl.expr.ArrayExpression, protein_coding: bool
    ) -> hl.expr.StructExpression:
        """
        Process VEP consequences to generate summary annotations.

        :param csq_list: VEP consequences list to be processed.
        :param protein_coding: Whether variant is in a protein-coding transcript.
        :return: Struct containing summary annotations.
        """
        lof = hl.null(hl.tstr)
        no_lof_flags = hl.null(hl.tbool)
        if protein_coding:
            all_lofs = csq_list.map(lambda x: x.lof)
            lof = hl.literal(loftee_labels).find(lambda x: all_lofs.contains(x))
            csq_list = hl.if_else(
                hl.is_defined(lof), csq_list.filter(lambda x: x.lof == lof), csq_list
            )
            no_lof_flags = hl.or_missing(
                hl.is_defined(lof),
                csq_list.any(lambda x: (x.lof == lof) & hl.is_missing(x.lof_flags)),
            )
        all_csq_terms = csq_list.flatmap(lambda x: x.consequence_terms)
        most_severe_csq = hl.literal(csq_order).find(
            lambda x: all_csq_terms.contains(x)
        )
        return hl.struct(
            most_severe_csq=most_severe_csq,
            protein_coding=protein_coding,
            lof=lof,
            no_lof_flags=no_lof_flags,
        )

    protein_coding = ht.vep.transcript_consequences.filter(
        lambda x: x.biotype == "protein_coding"
    )
    return ht.annotate(
        **hl.case(missing_false=True)
        .when(hl.len(protein_coding) > 0, _get_most_severe_csq(protein_coding, True))
        .when(
            hl.len(ht.vep.transcript_consequences) > 0,
            _get_most_severe_csq(ht.vep.transcript_consequences, False),
        )
        .when(
            hl.len(ht.vep.regulatory_feature_consequences) > 0,
            _get_most_severe_csq(ht.vep.regulatory_feature_consequences, False),
        )
        .when(
            hl.len(ht.vep.motif_feature_consequences) > 0,
            _get_most_severe_csq(ht.vep.motif_feature_consequences, False),
        )
        .default(_get_most_severe_csq(ht.vep.intergenic_consequences, False))
    )


def filter_vep_transcript_csqs(
    t: Union[hl.Table, hl.MatrixTable],
    vep_root: str = "vep",
    synonymous: bool = True,
    canonical: bool = True,
    ensembl_only: bool = True,
    filter_empty_csq: bool = True,
    **kwargs,
) -> Union[hl.Table, hl.MatrixTable]:
    """
    Filter VEP transcript consequences based on specified criteria, and optionally filter to variants where transcript consequences is not empty after filtering.

    If `filter_empty_csq` parameter is set to True, the Table/MatrixTable is filtered
    to variants where 'transcript_consequences' within the VEP annotation is not empty
    after the specified filtering criteria is applied.

    .. note::

        By default, the Table/MatrixTable is filtered to variants where
        'transcript_consequences' within the VEP annotation is not empty after filtering
        to Ensembl canonical transcripts with a most severe consequence of
        'synonymous_variant'.

    :param t: Input Table or MatrixTable.
    :param vep_root: Root for VEP annotation. Default is 'vep'.
    :param synonymous: Whether to filter to variants where the most severe consequence
        is 'synonymous_variant'. Default is True.
    :param canonical: Whether to filter to only canonical transcripts. Default is True.
    :param ensembl_only: Whether to filter to only Ensembl transcripts. This option is
        useful for deduplicating transcripts that are the same between RefSeq and
        Emsembl. Default is True.
    :param filter_empty_csq: Whether to filter out rows where 'transcript_consequences'
        is empty, after filtering 'transcript_consequences' to the specified criteria.
        Default is True.
    :param kwargs: Filtering criteria to apply to the VEP transcript consequences using
        `filter_vep_transcript_csqs_expr`. See that function for more details.
    :return: Table or MatrixTable with VEP transcript consequences filtered.
    """
    is_mt = isinstance(t, hl.MatrixTable)
    vep_data = {
        vep_root: t[vep_root].annotate(
            transcript_consequences=filter_vep_transcript_csqs_expr(
                t[vep_root].transcript_consequences,
                synonymous=synonymous,
                canonical=canonical,
                ensembl_only=ensembl_only,
                **kwargs,
            )
        )
    }
    t = t.annotate_rows(**vep_data) if is_mt else t.annotate(**vep_data)

    if filter_empty_csq:
        transcript_csq_expr = t[vep_root].transcript_consequences
        filter_expr = hl.is_defined(transcript_csq_expr) & (
            hl.len(transcript_csq_expr) > 0
        )
        t = t.filter_rows(filter_expr) if is_mt else t.filter(filter_expr)

    return t


def filter_vep_transcript_csqs_expr(
    csq_expr: Union[hl.expr.StructExpression, hl.expr.ArrayExpression],
    synonymous: bool = False,
    canonical: bool = False,
    mane_select: bool = False,
    ensembl_only: bool = False,
    protein_coding: bool = False,
    loftee_labels: Optional[List[str]] = None,
    no_lof_flags: bool = False,
    csqs: Optional[List[str]] = None,
    keep_csqs: bool = True,
    genes: Optional[List[str]] = None,
    keep_genes: bool = True,
    match_by_gene_symbol: bool = False,
    additional_filtering_criteria: Optional[
        List[Union[hl.expr.BooleanExpression, Callable]]
    ] = None,
) -> Union[hl.expr.BooleanExpression, hl.expr.ArrayExpression]:
    """
    Filter VEP transcript consequences based on specified criteria, and optionally filter to variants where transcript consequences is not empty after filtering.

    .. note::

        If `csqs` is not None or `synonymous` is True, and 'most_severe_consequence'
        is not already annotated on the `csq_expr` elements, the most severe
        consequence will be added to the `csq_expr` for filtering.

    :param csq_expr: VEP transcript consequences StructExpression or ArrayExpression.
    :param synonymous: Whether to filter to variants where the most severe consequence
        is 'synonymous_variant'. Default is False.
    :param canonical: Whether to filter to only canonical transcripts. Default is False.
    :param mane_select: Whether to filter to only MANE Select transcripts. Default is
        False.
    :param ensembl_only: Whether to filter to only Ensembl transcripts. This option is
        useful for deduplicating transcripts that are the same between RefSeq and
        Emsembl. Default is False.
    :param protein_coding: Whether to filter to only protein-coding transcripts.
        Default is False.
    :param loftee_labels: List of LOFTEE labels to filter to. Default is None, which
        filters to all LOFTEE labels.
    :param no_lof_flags: Whether to filter to consequences with no LOFTEE flags.
        Default is False.
    :param csqs: Optional list of consequence terms to filter to. Transcript
        consequences are filtered to those where 'most_severe_consequence' is in the
        list of consequence terms `csqs`. Default is None.
    :param keep_csqs: Whether to keep transcript consequences that are in `csqs`. If
        set to False, transcript consequences that are in `csqs` will be removed.
        Default is True.
    :param genes: Optional list of genes to filter VEP transcript consequences to.
        Default is None.
    :param keep_genes: Whether to keep transcript consequences that are in `genes`. If
        set to False, transcript consequences that are in `genes` will be removed.
        Default is True.
    :param match_by_gene_symbol: Whether to match values in `genes` to VEP transcript
        consequences by 'gene_symbol' instead of 'gene_id'. Default is False.
    :param additional_filtering_criteria: Optional list of additional filtering
        criteria to apply to the VEP transcript consequences.
    :return: BooleanExpression indicating whether the consequence should be filtered
        or an ArrayExpression of the filtered VEP transcript consequences.
    """
    is_struct = isinstance(csq_expr, hl.expr.StructExpression)
    if synonymous:
        logger.info("Filtering to most severe consequence of synonymous_variant...")
        csqs = ["synonymous_variant"]

    csq_fields = csq_expr if is_struct else csq_expr.dtype.element_type.fields
    if csqs is not None:
        if "most_severe_consequence" not in csq_fields:
            logger.info("Adding most_severe_consequence annotation...")
            csq_expr = add_most_severe_consequence_to_consequence(csq_expr)

    def _filter_vep_csq_expr(
        csq: hl.expr.StructExpression,
        additional_filtering_criteria: Optional[List[hl.expr.BooleanExpression]] = None,
    ) -> hl.expr.BooleanExpression:
        """
        Filter VEP consequence StructExpression based on specified criteria.

        :param csq: VEP consequence StructExpression.
        :param additional_filtering_criteria: Optional list of additional filtering.
        :return: BooleanExpression for filtering VEP consequence StructExpression.
        """
        criteria = hl.bool(True)
        if csqs is not None:
            found = hl.literal(csqs).contains(csq.most_severe_consequence)
            if keep_csqs:
                criteria &= found
            else:
                criteria &= ~found
        if canonical:
            logger.info("Filtering to canonical transcripts")
            criteria &= csq.canonical == 1
        if mane_select:
            logger.info("Filtering to MANE Select transcripts...")
            criteria &= hl.is_defined(csq.mane_select)
        if ensembl_only:
            logger.info("Filtering to Ensembl transcripts...")
            criteria &= csq.transcript_id.startswith("ENST")
        if protein_coding:
            logger.info("Filtering to protein coding transcripts...")
            criteria &= csq.biotype == "protein_coding"
        if loftee_labels:
            logger.info(
                "Filtering to consequences with LOFTEE labels: %s...", loftee_labels
            )
            criteria &= hl.set(loftee_labels).contains(csq.lof)
        if no_lof_flags:
            logger.info("Filtering to consequences with no LOFTEE flags...")
            criteria &= hl.is_missing(csq.lof_flags) | (csq.lof_flags == "")
        if genes is not None:
            logger.info("Filtering to genes of interest...")
            gene_field = "gene_symbol" if match_by_gene_symbol else "gene_id"
            found = hl.literal(genes).contains(csq[gene_field])
            if keep_genes:
                criteria &= found
            else:
                criteria &= ~found
        if additional_filtering_criteria is not None:
            logger.info("Filtering to variants with additional criteria...")
            criteria &= functools.reduce(operator.iand, additional_filtering_criteria)

        return criteria

    if is_struct:
        return _filter_vep_csq_expr(csq_expr, additional_filtering_criteria)
    else:
        return csq_expr.filter(
            lambda x: _filter_vep_csq_expr(
                x,
                (
                    [f(x) for f in additional_filtering_criteria]
                    if additional_filtering_criteria is not None
                    else None
                ),
            )
        )


def add_most_severe_csq_to_tc_within_vep_root(
    t: Union[hl.Table, hl.MatrixTable], vep_root: str = "vep"
) -> Union[hl.Table, hl.MatrixTable]:
    """
    Add most_severe_consequence annotation to 'transcript_consequences' within the vep root annotation.

    :param t: Input Table or MatrixTable.
    :param vep_root: Root for vep annotation (probably vep).
    :return: Table or MatrixTable with most_severe_consequence annotation added.
    """
    annotation = t[vep_root].annotate(
        transcript_consequences=t[vep_root].transcript_consequences.map(
            add_most_severe_consequence_to_consequence
        )
    )
    return (
        t.annotate_rows(**{vep_root: annotation})
        if isinstance(t, hl.MatrixTable)
        else t.annotate(**{vep_root: annotation})
    )


def explode_by_vep_annotation(
    t: Union[hl.Table, hl.MatrixTable],
    vep_annotation: str = "transcript_consequences",
    vep_root: str = "vep",
) -> Union[hl.Table, hl.MatrixTable]:
    """
    Explode the specified VEP annotation on the input Table/MatrixTable.

    :param t: Input Table or MatrixTable.
    :param vep_annotation: Name of annotation in `vep_root` to explode.
    :param vep_root: Name used for root VEP annotation. Default is 'vep'.
    :return: Table or MatrixTable with exploded VEP annotation.
    """
    if vep_annotation not in t[vep_root].keys():
        raise ValueError(
            f"{vep_annotation} is not a row field of the {vep_root} annotation in"
            " Table/MatrixTable!"
        )
    # Create top-level annotation for `vep_annotation` and explode it.
    if isinstance(t, hl.Table):
        t = t.transmute(**{vep_annotation: t[vep_root][vep_annotation]})
        t = t.explode(t[vep_annotation])
    else:
        t = t.transmute_rows(**{vep_annotation: t[vep_root][vep_annotation]})
        t = t.explode_rows(t[vep_annotation])

    return t


def get_loftee_end_trunc_filter_expr(
    csq_expr: hl.expr.StructExpression,
    gerp_dist_cutoff: float = 0.0,
) -> hl.expr.BooleanExpression:
    """
    Get the expression for LOFTEE's END_TRUNC filter based on the GERP distance cutoff.

    The end truncation filter is based on the GERP distance cutoff (`gerp_dist_cutoff`)
    and the 'GERP_DIST' and '50_BP_RULE' annotations in the LOFTEE annotations.

    True is returned if the GERP distance is less than the cutoff and the '50_BP_RULE'
    annotation is not 'PASS'.

    :param csq_expr: StructExpression containing the LOFTEE annotation 'lof_info', with
        'GERP_DIST' and '50_BP_RULE' info.
    :param gerp_dist_cutoff: GERP distance cutoff for end truncation. Default is 0.0.
    :return: BooleanExpression for end truncation annotation.
    """
    lof_info_expr = hl.dict(
        csq_expr.lof_info.split(",")
        .map(lambda x: x.split(":"))
        .map(lambda x: (x[0], hl.or_missing(x.length() > 1, x[1])))
    )

    end_trunc_expr = hl.or_else(
        hl.float64(lof_info_expr.get("GERP_DIST", "0")) < gerp_dist_cutoff, False
    ) & hl.or_else(lof_info_expr.get("50_BP_RULE", "") != "PASS", False)

    return end_trunc_expr


def update_loftee_end_trunc_filter(
    csq_expr: Union[hl.expr.StructExpression, hl.expr.ArrayExpression],
    gerp_dist_cutoff: float = 0.0,
) -> hl.expr.StructExpression:
    """
    Update the LOFTEE end truncation filter in the input Struct or Array of Structs.

    The LOFTEE end truncation filter is updated based on the GERP distance cutoff
    (`gerp_dist_cutoff`) using `get_loftee_end_trunc_filter_expr`.

    The 'lof_filter' field in the input Struct or Array of Structs is updated to include
    'END_TRUNC' if the end truncation filter is met, and 'END_TRUNC' is removed if the
    end truncation filter is not met.

    Then the 'lof' field in the input Struct or Array of Structs is updated to 'HC' if
    the new 'lof_filter' is missing, and 'LC' if it's not missing.

    :param csq_expr: Struct or Array of Structs containing the LOFTEE annotations.
    :param gerp_dist_cutoff: GERP distance cutoff for end truncation. Default is 0.0.
    :return: Struct or Array of Structs with updated LOFTEE end truncation filter
        annotation.
    """

    def _update_csq_struct(csq_expr: hl.expr.StructExpression):
        """
        Update the LOFTEE end truncation filter in the input Struct.

        :param csq_expr: Struct containing the LOFTEE annotations.
        :return: Consequence Struct with updated LOFTEE annotations.
        """
        end_trunc_expr = get_loftee_end_trunc_filter_expr(csq_expr, gerp_dist_cutoff)
        filter_expr = hl.or_else(
            hl.set(csq_expr.lof_filter.split(",")), hl.empty_set(hl.tstr)
        )
        filter_expr = hl.if_else(
            end_trunc_expr,
            filter_expr.add("END_TRUNC"),
            filter_expr.remove("END_TRUNC"),
        )
        filter_expr = hl.or_missing(filter_expr.length() > 0, hl.delimit(filter_expr))

        lof_expr = hl.or_missing(
            hl.is_defined(csq_expr.lof),
            hl.if_else(hl.is_missing(filter_expr), "HC", "LC"),
        )

        return hl.struct(lof_filter=filter_expr, lof=lof_expr)

    if isinstance(csq_expr, hl.expr.StructExpression):
        return csq_expr.annotate(**_update_csq_struct(csq_expr))
    else:
        return csq_expr.map(lambda x: x.annotate(**_update_csq_struct(x)))

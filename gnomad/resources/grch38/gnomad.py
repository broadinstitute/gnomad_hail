# noqa: D100

import logging
from typing import Optional, Union

import hail as hl

from gnomad.resources.resource_utils import (
    DataException,
    GnomadPublicMatrixTableResource,
    GnomadPublicTableResource,
    VersionedMatrixTableResource,
    VersionedTableResource,
)
from gnomad.sample_qc.ancestry import POP_NAMES
from gnomad.utils.annotations import add_gks_va, add_gks_vrs

logging.basicConfig(
    format="%(asctime)s (%(name)s %(lineno)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CURRENT_EXOME_RELEASE = ""
CURRENT_GENOME_RELEASE = "3.1.2"
CURRENT_GENOME_COVERAGE_RELEASE = "3.0.1"
EXOME_RELEASES = []
GENOME_RELEASES = ["3.0", "3.1", "3.1.1", "3.1.2"]
GENOME_COVERAGE_RELEASES = GENOME_RELEASES + ["3.0.1"]
DATA_TYPES = ["genomes"]
MAJOR_RELEASES = ["v3", "v4"]
CURRENT_MAJOR_RELEASE = MAJOR_RELEASES[-1]


GENOME_POPS = ["AFR", "AMI", "AMR", "ASJ", "EAS", "FIN", "NFE", "SAS", "OTH"]
SUBSETS = {
    "v3": [
        "non_v2",
        "non_topmed",
        "non_cancer",
        "controls_and_biobanks",
        "non_neuro",
        "tgp",
        "hgdp",
    ],
    "v4": ["ukb", "non_ukb", "non_topmed"],
}
"""
Order to sort subgroupings during VCF export by version.

Ensures that INFO labels in VCF are in desired order (e.g., tgp_raw_AC_esn_XX).
"""

GROUPS = ["adj", "raw"]
"""
Group names used to generate labels for high quality genotypes and all raw genotypes.

Used in VCF export.
"""

SEXES = ["XX", "XY"]
"""
Sample sexes used in VCF export.

Used to stratify frequency annotations (AC, AN, AF) for each sex.
"""

POPS = {
    "v3": ["afr", "ami", "amr", "asj", "eas", "fin", "nfe", "oth", "sas", "mid"],
    "v4": [
        "afr",
        "amr",
        "asj",
        "eas",
        "fin",
        "mid",
        "nfe",
        "remaining",
        "sas",
    ],
}
"""
Global ancestry groups in gnomAD by version.
"""

COHORTS_WITH_POP_STORED_AS_SUBPOP = ["tgp", "hgdp"]
"""
Subsets in gnomAD v3.1 that are broken down by their known subpops instead of global pops in the frequency struct.
"""

TGP_POPS = [
    "esn",
    "pur",
    "pjl",
    "clm",
    "jpt",
    "chb",
    "stu",
    "itu",
    "tsi",
    "mxl",
    "ceu",
    "msl",
    "yri",
    "beb",
    "fin",
    "khv",
    "cdx",
    "lwk",
    "acb",
    "asw",
    "ibs",
    "gbr",
    "pel",
    "gih",
    "chs",
    "gwd",
]
"""
1000 Genomes Project (1KG/TGP) subpops.
"""

HGDP_POPS = [
    "japanese",
    "papuan",
    "adygei",
    "orcadian",
    "biakapygmy",
    "yakut",
    "han",
    "uygur",
    "miaozu",
    "mongola",
    "balochi",
    "bedouin",
    "russian",
    "daur",
    "pima",
    "hezhen",
    "sindhi",
    "yizu",
    "oroqen",
    "san",
    "tuscan",
    "tu",
    "palestinian",
    "tujia",
    "druze",
    "pathan",
    "basque",
    "makrani",
    "italian",
    "naxi",
    "karitiana",
    "sardinian",
    "mbutipygmy",
    "mozabite",
    "yoruba",
    "lahu",
    "dai",
    "cambodian",
    "melanesian",
    "french",
    "brahui",
    "hazara",
    "bantusafrica",
    "surui",
    "mandenka",
    "kalash",
    "xibo",
    "colombian",
    "bantukenya",
    "she",
    "burusho",
    "maya",
]
"""
Human Genome Diversity Project (HGDP) subpops.
"""

TGP_POP_NAMES = {
    "chb": "Han Chinese",
    "jpt": "Japanese",
    "chs": "Southern Han Chinese",
    "cdx": "Chinese Dai",
    "khv": "Kinh",
    "ceu": "Utah Residents (European Ancestry)",
    "tsi": "Toscani",
    "fin": "Finnish",
    "gbr": "British",
    "ibs": "Iberian",
    "yri": "Yoruba",
    "lwk": "Luhya",
    "gwd": "Gambian",
    "msl": "Mende",
    "esn": "Esan",
    "asw": "African-American",
    "acb": "African Caribbean",
    "mxl": "Mexican-American",
    "pur": "Puerto Rican",
    "clm": "Colombian",
    "pel": "Peruvian",
    "gih": "Gujarati",
    "pjl": "Punjabi",
    "beb": "Bengali",
    "stu": "Sri Lankan Tamil",
    "itu": "Indian Telugu",
}
"""
1000 Genomes Project (1KG/TGP) pop label map.
"""

POPS_STORED_AS_SUBPOPS = TGP_POPS + HGDP_POPS
POPS_TO_REMOVE_FOR_POPMAX = {"asj", "fin", "oth", "ami", "mid", "remaining"}
"""
Populations that are removed before popmax calculations.
"""

DOWNSAMPLINGS = {
    "v3": [
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
        40000,
        50000,
        60000,
        70000,
        75000,
        80000,
        85000,
        90000,
        95000,
        100000,
        110000,
        120000,
    ],
    "v4": [
        10,
        100,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
    ],
}
"""
List of the downsampling numbers to use for frequency calculations by version.
"""

gnomad_syndip = VersionedMatrixTableResource(
    default_version="3.0",
    versions={
        "3.0": GnomadPublicMatrixTableResource(
            path="gs://gnomad-public-requester-pays/truth-sets/hail-0.2/gnomad_v3_syndip.b38.mt"
        )
    },
)

na12878 = VersionedMatrixTableResource(
    default_version="3.0",
    versions={
        "3.0": GnomadPublicMatrixTableResource(
            path="gs://gnomad-public-requester-pays/truth-sets/hail-0.2/gnomad_v3_na12878.mt"
        )
    },
)


def get_coverage_ht(
    coverage_ht: Union[str, hl.Table], data_type: str, coverage_version: str
):
    """
    Load a coverage hail table if needed.

    If coverage_ht is 'auto', loads the default coverage table for the
    data_type and coverage_version. If it's already a hail table, return it.
    Otherwise return None.

    :param coverage_ht: a hail table, or 'auto' (otherwise return None).
    :param data_type: a gnomad dataset type, as in 'genomes' or 'exomes'
    :param coverage_version: gnomad release version the coverage table is built on
    :return: hail table with coverage info, or None
    """
    if coverage_ht == "auto":
        return hl.read_table(coverage(data_type).versions[coverage_version].path)
    elif isinstance(coverage_ht, hl.Table):
        return coverage_ht
    else:
        return None


def _public_release_ht_path(data_type: str, version: str) -> str:
    """
    Get public release table path.

    :param data_type: One of "exomes" or "genomes"
    :param version: One of the release versions of gnomAD on GRCh38
    :return: Path to release Table
    """
    version_prefix = "r" if version.startswith("3.0") else "v"
    return f"gs://gnomad-public-requester-pays/release/{version}/ht/{data_type}/gnomad.{data_type}.{version_prefix}{version}.sites.ht"


def _public_coverage_ht_path(data_type: str, version: str) -> str:
    """
    Get public coverage hail table.

    :param data_type: One of "exomes" or "genomes"
    :param version: One of the release versions of gnomAD on GRCh38
    :return: path to coverage Table
    """
    version_prefix = "r" if version.startswith("3.0") else "v"
    return f"gs://gnomad-public-requester-pays/release/{version}/coverage/{data_type}/gnomad.{data_type}.{version_prefix}{version}.coverage.ht"


def public_release(data_type: str) -> VersionedTableResource:
    """
    Retrieve publicly released versioned table resource.

    :param data_type: One of "exomes" or "genomes"
    :return: Release Table
    """
    if data_type not in DATA_TYPES:
        raise DataException(
            f"{data_type} not in {DATA_TYPES}, please select a data type from"
            f" {DATA_TYPES}"
        )

    if data_type == "exomes":
        current_release = CURRENT_EXOME_RELEASE
        releases = EXOME_RELEASES
    else:
        current_release = CURRENT_GENOME_RELEASE
        releases = GENOME_RELEASES

    return VersionedTableResource(
        current_release,
        {
            release: GnomadPublicTableResource(
                path=_public_release_ht_path(data_type, release)
            )
            for release in releases
        },
    )


def coverage(data_type: str) -> VersionedTableResource:
    """
    Retrieve gnomAD's coverage table by data_type.

    :param data_type: One of "exomes" or "genomes"
    :return: Coverage Table
    """
    if data_type not in DATA_TYPES:
        raise DataException(
            f"{data_type} not in {DATA_TYPES}, please select a data type from"
            f" {DATA_TYPES}"
        )

    if data_type == "exomes":
        current_release = CURRENT_EXOME_RELEASE
        releases = EXOME_RELEASES
    else:
        current_release = CURRENT_GENOME_COVERAGE_RELEASE
        releases = GENOME_COVERAGE_RELEASES

    return VersionedTableResource(
        current_release,
        {
            release: GnomadPublicTableResource(
                path=_public_coverage_ht_path(data_type, release)
            )
            for release in releases
        },
    )


def coverage_tsv_path(data_type: str, version: Optional[str] = None) -> str:
    """
    Retrieve gnomAD's coverage table by data_type.

    :param data_type: One of "exomes" or "genomes"
    :return: Coverage Table
    """
    if data_type not in DATA_TYPES:
        raise DataException(
            f"{data_type} not in {DATA_TYPES}, please select a data type from"
            f" {DATA_TYPES}"
        )

    if data_type == "exomes":
        if version is None:
            version = CURRENT_EXOME_RELEASE
        elif version not in EXOME_RELEASES:
            raise DataException(
                f"Version {version} of gnomAD exomes for GRCh38 does not exist"
            )
    else:
        if version is None:
            version = CURRENT_GENOME_COVERAGE_RELEASE
        elif version not in GENOME_COVERAGE_RELEASES:
            raise DataException(
                f"Version {version} of gnomAD genomes for GRCh38 does not exist"
            )

    version_prefix = "r" if version.startswith("3.0") else "v"
    return f"gs://gcp-public-data--gnomad/release/{version}/coverage/{data_type}/gnomad.{data_type}.{version_prefix}{version}.coverage.summary.tsv.bgz"


def release_vcf_path(data_type: str, version: str, contig: str) -> str:
    """
    Publically released VCF. Provide specific contig, i.e. "chr20", to retrieve contig specific VCF.

    :param data_type: One of "exomes" or "genomes"
    :param version: One of the release versions of gnomAD on GRCh37
    :param contig: Single contig "chr1" to "chrY"
    :return: Path to VCF
    """
    if version.startswith("2"):
        raise DataException(
            f"gnomAD version {version} is not available on reference genome GRCh38"
        )

    contig = f".{contig}" if contig else ""
    version_prefix = "r" if version.startswith("3.0") else "v"
    return f"gs://gcp-public-data--gnomad/release/{version}/vcf/{data_type}/gnomad.{data_type}.{version_prefix}{version}.sites{contig}.vcf.bgz"


def gnomad_gks_batch(
    locus_interval: hl.IntervalExpression,
    version: str,
    data_type: str = "genomes",
    by_ancestry_group: bool = False,
    by_sex: bool = False,
    vrs_only: bool = False,
    custom_ht: hl.Table = None,
    skip_coverage: bool = False,
    custom_coverage_ht: hl.Table = None,
) -> list:
    """
    Perform gnomad GKS annotations on a range of variants at once.

    :param locus_interval: Hail IntervalExpression of locus<reference_genome>.
        e.g. hl.locus_interval('chr1', 1, 50000000, reference_genome="GRCh38")
    :param version: String of version of gnomAD release to use.
    :param data_type: String of either "exomes" or "genomes" for the type of reads that are desired.
    :param by_ancestry_group: Boolean to pass for frequency information for each cohort.
    :param by_sex: Boolean to pass to return freq info for each cohort split by chromosomal sex.
    :param vrs_only: Boolean to pass for only VRS info to be returned
        (will not include allele frequency information).
    :param custom_ht: Table to use instead of return from public_release() method.
    :param skip_coverage: Bool to pass to skip adding coverage stats.
    :param custom_coverage_ht: Custom table to use for coverage stats if not release coverage table.
    :return: List of dictionaries containing VRS information
        (and freq info split by ancestry groups and sex if desired) for specified variant.
    """
    # Read public_release table if no custom table provided
    if custom_ht:
        ht = custom_ht
    else:
        ht = hl.read_table(public_release(data_type).versions[version].path)

    high_level_version = f"v{version.split('.')[0]}"

    # Read coverage statistics if requested
    if high_level_version == "v3":
        coverage_version = "3.0.1"
    else:
        raise NotImplementedError(
            "gnomad_gks() is currently only implemented for gnomAD v3."
        )

    coverage_ht = None

    if not skip_coverage:
        if custom_coverage_ht:
            coverage_ht = custom_coverage_ht
        else:
            coverage_ht = hl.read_table(
                coverage("genomes").versions[coverage_version].path
            )

    # Retrieve ancestry groups from the imported POPS dictionary.
    pops_list = list(POPS[high_level_version]) if by_ancestry_group else None

    # Throw warnings if contradictory arguments are passed.
    if by_ancestry_group and vrs_only:
        logger.warning(
            "Both 'vrs_only' and 'by_ancestry_groups' have been specified. Ignoring"
            " 'by_ancestry_groups' list and returning only VRS information."
        )
    elif by_sex and not by_ancestry_group:
        logger.warning(
            "Splitting whole database by sex is not yet supported. If using 'by_sex',"
            " please also specify 'by_ancestry_group' to stratify by."
        )

    # Call and return add_gks*() for chosen arguments.

    # Filter to interval before adding annotations
    ht = hl.filter_intervals(ht, [locus_interval])

    # Collect all variants as structs, so all dictionary construction can be
    # done in native Python
    variant_list = ht.collect()

    # Assemble output dicts with VRS and optionally frequency, append to list,
    # then return list
    outputs = []
    for variant in variant_list:
        vrs_variant = add_gks_vrs(variant.locus, variant.info.vrs)

        out = {
            "locus": {
                "contig": variant.locus.contig,
                "position": variant.locus.position,
                "reference_genome": variant.locus.reference_genome.name,
            },
            "alleles": variant.alleles,
            "gks_vrs_variant": vrs_variant,
        }

        if not vrs_only:
            va_freq_dict = add_gks_va(
                input_dict=variant,
                label_name="gnomAD",
                label_version=version,
                coverage_ht=coverage_ht,
                ancestry_groups=pops_list,
                ancestry_groups_dict=POP_NAMES,
                by_sex=by_sex,
                frequency_index=ht.freq_index_dict.collect()[0],
            )

            # Assign existing VRS information to "focusAllele" key
            va_freq_dict["focusAllele"] = vrs_variant
            out["gks_va_freq"] = va_freq_dict

        # Append variant dictionary to list of outputs
        outputs.append(out)

    return outputs

from hail import *

CURRENT_HAIL_VERSION = "0.1"
CURRENT_RELEASE = "2.0.2"
CURRENT_GENOME_META = "2017-06-02"  # YYYY-MM-DD
CURRENT_EXOME_META = "2017-06-02"

RELEASES = ["2.0.1", "2.0.2"]

GENOME_POPS = ['AFR', 'AMR', 'ASJ', 'EAS', 'FIN', 'NFE', 'OTH']
EXOME_POPS = ['AFR', 'AMR', 'ASJ', 'EAS', 'FIN', 'NFE', 'OTH', 'SAS']
EXAC_POPS = ["AFR", "AMR", "EAS", "FIN", "NFE", "OTH", "SAS"]


def public_exomes_vds_path(split=False, version=CURRENT_RELEASE):
    return 'gs://gnomad-public/release/{0}/vds/exomes/gnomad.exomes.r{0}.sites{1}.vds'.format(version, ".split" if split else "")


def public_genomes_vds_path(split=False, version=CURRENT_RELEASE):
    return 'gs://gnomad-public/release/{0}/vds/genomes/gnomad.genomes.r{0}.sites{1}.vds'.format(version, ".split" if split else "")


def get_gnomad_data(hc, data_type, hardcalls=None, split=False, hail_version=CURRENT_HAIL_VERSION,
                    meta_version=None, meta_root='sa.meta', sample_filtering=None,
                    duplicate_mapping_root=None, fam_root='sa.fam', vqsr=True):
    """
    Wrapper function to get gnomAD data as VDS.

    :param HailContext hc: HailContext
    :param str data_type: One of `exomes` or `genomes`
    :param str hardcalls: One of `adj` or `raw` if hardcalls are desired (leave as None for raw data)
    :param bool split: Whether the dataset should be split (only applies to hardcalls)
    :param str hail_version: One of the HAIL_VERSIONs
    :param str meta_version: Version of metadata (None for current)
    :param str meta_root: Where to put metadata. Set to None if no metadata is desired.
    :param str sample_filtering: Whether to filter samples - can be one of: None (all samples), release, qc, or a custom sa.meta.X == "Y" expression.
    :param str duplicate_mapping_root: Where to put the duplicate genome/exome samples ID mapping (default is None -- do not annotate)
    :param str fam_root: Where to put the pedigree information. Set to None if no pedigree information is desired.
    :param bool vqsr: Whether to add VQSR information for exomes (goes into va.info)
    :return: Chosen VDS
    :rtype: VariantDataset
    """
    vds = hc.read(get_gnomad_data_path(data_type, hardcalls=hardcalls, split=split, hail_version=hail_version))

    if meta_root:
        vds = vds.annotate_samples_table(get_gnomad_meta(hc, data_type, meta_version), root=meta_root)

    if sample_filtering:
        if sample_filtering == 'release':
            vds = vds.filter_samples_expr('sa.meta.keep')
        elif sample_filtering == 'qc':
            vds = vds.filter_samples_expr('sa.meta.qc_sample')
        else:
            vds = vds.filter_samples_expr(sample_filtering)

    if duplicate_mapping_root:
        vds = vds.annotate_samples_table(
            hc.import_table(genomes_exomes_duplicate_ids_tsv_path,
                            impute=True,
                            key='exome_id' if data_type == "exomes" else 'genome_id'),
            root=duplicate_mapping_root)

    if fam_root:
        vds = vds.annotate_samples_table(
            KeyTable.import_fam(exomes_fam_path if data_type == "exomes" else genomes_fam_path),
            root=fam_root
        )

    pops = EXOME_POPS if data_type == 'exomes' else GENOME_POPS
    vds = vds.annotate_global('global.pops', map(lambda x: x.lower(), pops), TArray(TString()))

    if data_type == 'exomes' and vqsr:
        vqsr_vds = hc.read(vqsr_exomes_sites_vds_path())
        annotations = ['culprit', 'POSITIVE_TRAIN_SITE', 'NEGATIVE_TRAIN_SITE', 'VQSLOD']
        vds = vds.annotate_variants_vds(vqsr_vds, expr=', '.join(['va.info.%s = vds.info.%s' % (a, a) for a in annotations]))

    return vds


def get_gnomad_meta(hc, data_type, version=None):
    """
    Wrapper function to get gnomAD metadata as keytable

    :param HailContext hc: HailContext
    :param str data_type: One of `exomes` or `genomes`
    :param str version: Metadata version (None for current)
    :return: Metadata KeyTable
    :rtype: KeyTable
    """
    return (
        hc
        .import_table(get_gnomad_meta_path(data_type, version), impute=True)
        .key_by("sample" if data_type == "exomes" else "Sample")
    )


def get_gnomad_data_path(data_type, hardcalls=None, split=False, hail_version=CURRENT_HAIL_VERSION):
    """
    Wrapper function to get paths to gnomAD data

    :param str data_type: One of `exomes` or `genomes`
    :param str hardcalls: One of `adj` or `raw` if hardcalls are desired (leave as None for raw data)
    :param bool split: Whether the dataset should be split (only applies to hardcalls)
    :param str hail_version: One of the HAIL_VERSIONs
    :return: Path to chosen VDS
    :rtype: str
    """
    if hardcalls is not None and hardcalls not in ('adj', 'raw'):
        return DataException("Select hardcalls of 'adj', 'raw', or None")
    if data_type == 'exomes':
        if not hardcalls:
            return raw_exomes_vds_path(hail_version)
        else:
            return hardcalls_exomes_vds_path(split, hardcalls == 'adj', hail_version)
    elif data_type == 'genomes':
        if not hardcalls:
            return raw_genomes_vds_path(hail_version)
        else:
            return hardcalls_genomes_vds_path(split, hardcalls == 'adj', hail_version)
    return DataException("Select data_type of 'genomes' or 'exomes'")


def get_gnomad_meta_path(data_type, version=None):
    """
    Wrapper function to get paths to gnomAD metadata

    :param str data_type: One of `exomes` or `genomes`
    :param str version: String with version (date) for metadata
    :return: Path to chosen metadata file
    :rtype: str
    """
    if data_type == 'exomes':
        if version:
            return metadata_exomes_tsv_path(version)
        return metadata_exomes_tsv_path()
    elif data_type == 'genomes':
        if version:
            return metadata_genomes_tsv_path(version)
        return metadata_genomes_tsv_path()
    return DataException("Select data_type of 'genomes' or 'exomes'")


def vqsr_exomes_sites_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad/raw/hail-{0}/vds/exomes/gnomad.exomes.vqsr.sites.vds'.format(hail_version)


def raw_exomes_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad/raw/hail-{0}/vds/exomes/gnomad.exomes.vds'.format(hail_version)


def raw_genomes_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad/raw/hail-{0}/vds/genomes/gnomad.genomes.vds'.format(hail_version)


def raw_exac_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad/raw/hail-{0}/vds/exac/exac.vds'.format(hail_version)


def hardcalls_exomes_vds_path(split=False, adj=False, hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad/hardcalls/hail-{0}/vds/exomes/gnomad.exomes.{1}{2}.vds'.format(hail_version,
                                                                                       "adj" if adj else "raw",
                                                                                       ".split" if split else "")


def hardcalls_genomes_vds_path(split=False, adj=False, hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad/hardcalls/hail-{0}/vds/genomes/gnomad.genomes.{1}{2}.vds'.format(hail_version,
                                                                                         "adj" if adj else "raw",
                                                                                         ".split" if split else "")

gnomad_pca_vds_path = "gs://gnomad-genomes/sampleqc/gnomad.pca.vds"


def metadata_genomes_tsv_path(version=CURRENT_GENOME_META):
    return 'gs://gnomad/metadata/genomes/gnomad.genomes.metadata.{0}.tsv.bgz'.format(version)


def metadata_exomes_tsv_path(version=CURRENT_EXOME_META):
    return 'gs://gnomad/metadata/exomes/gnomad.exomes.metadata.{0}.tsv.bgz'.format(version)


genomes_fam_path = "gs://gnomad/metadata/genomes/gnomad.genomes.fam"
exomes_fam_path = "gs://gnomad/metadata/exomes/gnomad.exomes.fam"
genomes_exomes_duplicate_ids_tsv_path = "gs://gnomad/metadata/genomes_exomes_duplicate_ids.tsv"


def omni_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad-public/truth-sets/hail-{0}/1000G_omni2.5.b37.vds'.format(hail_version)


def mills_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad-public/truth-sets/hail-{0}/Mills_and_1000G_gold_standard.indels.b37.vds'.format(hail_version)


def hapmap_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad-public/truth-sets/hail-{0}/hapmap_3.3.b37.vds'.format(hail_version)


def kgp_high_conf_snvs_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad-public/truth-sets/hail-{0}/1000G_phase1.snps.high_confidence.b37.vds'.format(hail_version)


def NA12878_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad-public/truth-sets/hail-{0}/NA12878_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-Solid-10X_CHROM1-X_v3.3_highconf.vds'.format(hail_version)


def syndip_vds_path(hail_version=CURRENT_HAIL_VERSION):
    return 'gs://gnomad-public/truth-sets/hail-{0}/hybrid.m37m.vds'.format(hail_version)


dbsnp_vcf_path = "gs://gnomad-public/truth-sets/source/All_20160601.vcf.bgz"

NA12878_high_conf_regions_bed_path = "gs://gnomad-public/truth-sets/source/NA12878_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-Solid-10X_CHROM1-X_v3.3_highconf.bed"
NA12878_high_conf_exome_regions_bed_path = "gs://gnomad-public/truth-sets/source/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed"
syndip_high_conf_regions_bed_path = "gs://gnomad-public/truth-sets/source/hybrid.m37m.bed"
clinvar_tsv_path = "gs://gnomad-resources/annotations/clinvar_alleles.single.b37.tsv.gz"
clinvar_vds_path = "gs://gnomad-resources/annotations/clinvar_alleles.single.b37.vds"

# Useful intervals
lcr_intervals_path = "gs://gnomad-public/intervals/LCR.interval_list"
decoy_intervals_path = "gs://gnomad-public/intervals/mm-2-merged.bed.gz"
purcell5k_intervals_path = "gs://gnomad-public/intervals/purcell5k.interval_list"

# Exome intervals
exomes_high_conf_regions_intervals_path = "gs://gnomad-public/intervals/exomes_high_coverage.auto.interval_list"
exome_calling_intervals_path = 'gs://gnomad-public/intervals/exome_calling_regions.v1.interval_list'
evaluation_intervals_path = 'gs://gnomad-public/intervals/exome_evaluation_regions.v1.noheader.interval_list'
high_coverage_intervals_path = 'gs://gnomad-public/intervals/high_coverage.auto.interval_list'

vep_config = "/vep/vep-gcloud.properties"

# Annotations
methylation_kt_path = "gs://gnomad-resources/methylation.kt"
context_vds_path = 'gs://gnomad-resources/constraint/context_processed.vds'


class DataException(Exception):
    pass


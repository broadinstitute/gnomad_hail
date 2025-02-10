Variant Effect Predictor (VEP)
==============================

To use the `Ensembl Variant Effect Predictor <https://www.ensembl.org/vep>`_ with Hail on Google Dataproc,
the ``--vep`` flag must be included when starting the cluster. Note that a cluster's VEP configuration is
tied to a specific reference genome.

.. code-block:: shell

   hailctl dataproc start cluster-name --vep GRCh37 --packages gnomad

.. note::

   VEP data is stored in requester pays buckets. Reading from these buckets will bill charges to the project
   in which the cluster is created.

Import variants into a sites-only Hail Table::

   import hail as hl

   ds = hl.import_vcf("/path/to/data.vcf.gz", reference_genome="GRCh37", drop_samples=True).rows()

Annotate variants with VEP consequences::

   from gnomad.utils.vep import vep_or_lookup_vep

   ds = vep_or_lookup_vep(ds, reference="GRCh37")

:py:func:`vep_or_lookup_vep <gnomad.utils.vep.vep_or_lookup_vep>` uses a precomputed dataset to
drastically speed up this process.

Identify the most severe consequence for each variant::

   from gnomad.utils.vep import process_consequences

   ds = process_consequences(ds)

:py:func:`process_consequences <gnomad.utils.vep.process_consequences>` adds ``worst_consequence_term``,
``worst_csq_for_variant``, ``worst_csq_by_gene`` and other fields to ``ds.vep``.

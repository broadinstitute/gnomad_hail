Resource Sources
================

gnomAD data is available through `multiple cloud providers' public datasets programs <https://gnomad.broadinstitute.org/news/2020-10-open-access-to-gnomad-data-on-multiple-cloud-providers/>`_.

The functions in the :doc:`gnomad.resources </api_reference/resources/index>` package can be configured to load data from different sources.

If Hail determines that is is running in a cloud provider's Spark environment, resources will default to being read from that cloud provider's datasets program.
For example, resource will be read from Azure Open Datasets if Hail determines that it is running on an Azure HDInsight cluster.
Otherwise, resources will default to being read from Google Cloud Public Datasets.
This can be configured using the ``GNOMAD_DEFAULT_PUBLIC_RESOURCE_SOURCE`` environment variable.

To load resources from a different source (for example, the gnomAD project's public GCS bucket), use:

.. code-block:: python

    from gnomad.resources.config import gnomad_public_resource_configuration, GnomadPublicResourceSource

    gnomad_public_resource_configuration.source = GnomadPublicResourceSource.GNOMAD

To see all available public sources for gnomAD resources, use:

.. code-block:: python

    from gnomad.resources.config import GnomadPublicResourceSource

    list(GnomadPublicResourceSource)

.. note::

   The gnomAD project's bucket (``gs://gnomad-public-requester-pays``) is `requester pays <https://cloud.google.com/storage/docs/requester-pays>`_, meaning that charges for data access and transfer will be billed to your Google Cloud project.

   Clusters must be configured to read requester pays buckets during creation. For example,

   .. code-block::

      hailctl dataproc start cluster-name --packages gnomad --requester-pays-allow-buckets gnomad-public-requester-pays

Custom Sources
--------------

Alternatively, instead of using one of the pre-defined public sources, a custom source can be provided.

.. code-block:: python

    from gnomad.resources.config import gnomad_public_resource_configuration

    gnomad_public_resource_configuration.source = "gs://my-bucket/gnomad-resources"

Environment Configuration
-------------------------

The default source can be configured through the ``GNOMAD_DEFAULT_PUBLIC_RESOURCE_SOURCE`` environment variable. This variable can be set to either the name of one of the public datasets programs or the URL of a custom source.

Examples:

- ``GNOMAD_DEFAULT_PUBLIC_RESOURCE_SOURCE="Google Cloud Public Datasets"``
- ``GNOMAD_DEFAULT_PUBLIC_RESOURCE_SOURCE="gs://my-bucket/gnomad-resources"``

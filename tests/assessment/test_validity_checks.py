"""Tests for the validity_checks module."""

import logging
from io import StringIO

import hail as hl
import pytest

from gnomad.assessment.validity_checks import (
    check_global_and_row_annot_lengths,
    check_missingness_of_struct,
    check_raw_and_adj_callstats,
    check_sex_chr_metrics,
    flatten_missingness_struct,
    make_group_sum_expr_dict,
    sum_group_callstats,
    unfurl_array_annotations,
)


@pytest.fixture
def ht_for_check_missingness_of_struct() -> hl.Table:
    """Fixture to set up a Hail Table with the desired nested structure and data."""
    # Create test data with nested structs.
    data = [
        {
            "idx": 0,
            "s": hl.struct(
                a=1,
                b="value1",
                c=hl.struct(
                    d=[hl.missing(hl.tstr), hl.missing(hl.tstr)],
                    e="test1",
                    f={"v1", "v2"},
                ),
            ),
        },
        {
            "idx": 1,
            "s": hl.struct(
                a=2,
                b="value2",
                c=hl.struct(
                    d=["not missing", hl.missing(hl.tstr)],
                    e=hl.missing(hl.tstr),
                    f={"v3", hl.missing(hl.tstr)},
                ),
            ),
        },
        {
            "idx": 2,
            "s": hl.struct(
                a=3,
                b=hl.missing(hl.tstr),
                c=hl.struct(
                    d=hl.missing(hl.tarray(hl.tstr)),
                    e=hl.missing(hl.tstr),
                    f=hl.empty_set(hl.tstr),
                ),
            ),
        },
        {
            "idx": 3,
            "s": hl.struct(
                a=4,
                b="value3",
                c=hl.struct(d=["foo", "bar"], e="test2", f=hl.empty_set(hl.tstr)),
            ),
        },
        {
            "idx": 4,
            "s": hl.struct(
                a=5,
                b="value4",
                c=hl.struct(
                    d=hl.empty_array(hl.tstr), e="test3", f=hl.empty_set(hl.tstr)
                ),
            ),
        },
    ]

    # Convert data into a Hail table.
    ht = hl.Table.parallelize(
        data,
        hl.tstruct(
            idx=hl.tint32,
            s=hl.tstruct(
                a=hl.tint32,
                b=hl.tstr,
                c=hl.tstruct(d=hl.tarray(hl.tstr), e=hl.tstr, f=hl.tset(hl.tstr)),
            ),
        ),
    )

    return ht


def test_check_missingness_of_struct(
    ht_for_check_missingness_of_struct: hl.Table,
) -> None:
    """Test check_missingness_of_struct and flatten results."""
    ht = ht_for_check_missingness_of_struct

    # Apply check_missingness_of_struct aggregate results.
    metric_missingness = check_missingness_of_struct(ht.s, "s")
    missingness_struct = ht.aggregate(hl.struct(**metric_missingness))

    # Flatten the result struct into a dictionary.
    missingness_dict = flatten_missingness_struct(missingness_struct)

    # Define expected missingness results.
    expected_result = {
        "s.a": 0.0,
        "s.b": 0.2,
        "s.c.d": 0.6,
        "s.c.e": 0.4,
        "s.c.f": 0.6,
    }

    # Compare the results with the expected values.
    for key, expected_value in expected_result.items():
        assert missingness_dict[key] == expected_value, (
            f"Mismatch for {key}: "
            f"expected {expected_value}, got {missingness_dict[key]}"
        )


@pytest.fixture()
def ht_for_check_array_struct_missingness() -> hl.Table:
    """Fixture to set up a sample Hail Table with array<struct> fields and global index dictionary ."""
    # Define rows of the Table.
    data = [
        {
            "idx": 0,
            "freq": [
                {"AC": 5, "AF": 0.1, "AN_eas": 20, "AN_sas": 3},
                {"AC": 10, "AF": 0.05, "AN_eas": 5, "AN_sas": None},
            ],
        },
        {
            "idx": 1,
            "freq": [
                {"AC": 6, "AF": 0.08, "AN_eas": None, "AN_sas": 4},
                {"AC": 8, "AF": 0.50, "AN_eas": None, "AN_sas": None},
            ],
        },
        {
            "idx": 2,
            "freq": [
                {"AC": 65, "AF": 0.18, "AN_eas": None, "AN_sas": 2},
                {"AC": 88, "AF": 0.20, "AN_eas": None, "AN_sas": None},
            ],
        },
        {"idx": 3, "freq": [{"AC": 8, "AF": 0.08, "AN_eas": 16, "AN_sas": 2}, None]},
    ]

    # Create Table.
    ht = hl.Table.parallelize(
        data,
        hl.tstruct(
            idx=hl.tint32,
            freq=hl.tarray(
                hl.tstruct(
                    AC=hl.tint32, AF=hl.tfloat64, AN_eas=hl.tint32, AN_sas=hl.tint32
                )
            ),
        ),
    )

    # Define global annotation for freq_index_dict.
    freq_index_dict = {"adj": 0, "raw": 1}
    ht = ht.annotate_globals(freq_index_dict=freq_index_dict)

    # Unfurl indexed array annotations.
    annotations = unfurl_array_annotations(ht, {"freq": "freq_index_dict"})
    ht = ht.annotate(**annotations)

    return ht


def test_unfurl_array_annotations(
    ht_for_check_array_struct_missingness: hl.Table,
) -> None:
    """Test the unfurl_array_annotations function for all rows."""
    ht = ht_for_check_array_struct_missingness
    indexed_array_annotations = {"freq": "freq_index_dict"}

    # Call the unfurl_array_annotations function.
    result = unfurl_array_annotations(ht, indexed_array_annotations)

    # Define names of the expected keys after unfurling.
    expected_keys = {
        "AC_adj",
        "AF_adj",
        "AN_eas_adj",
        "AN_sas_adj",
        "AC_raw",
        "AF_raw",
        "AN_eas_raw",
        "AN_sas_raw",
    }
    assert (
        set(result.keys()) == expected_keys
    ), "Unfurled keys do not match expected keys."

    # Annotate table with unfurled fields.
    ht = ht.annotate(**result)
    rows = ht.collect()

    # Define the expected values for each unfurled annotation.
    expected_values = [
        {
            "AC_adj": 5,
            "AF_adj": 0.1,
            "AN_eas_adj": 20,
            "AN_sas_adj": 3,
            "AC_raw": 10,
            "AF_raw": 0.05,
            "AN_eas_raw": 5,
            "AN_sas_raw": None,
        },
        {
            "AC_adj": 6,
            "AF_adj": 0.08,
            "AN_eas_adj": None,
            "AN_sas_adj": 4,
            "AC_raw": 8,
            "AF_raw": 0.50,
            "AN_eas_raw": None,
            "AN_sas_raw": None,
        },
        {
            "AC_adj": 65,
            "AF_adj": 0.18,
            "AN_eas_adj": None,
            "AN_sas_adj": 2,
            "AC_raw": 88,
            "AF_raw": 0.20,
            "AN_eas_raw": None,
            "AN_sas_raw": None,
        },
        {
            "AC_adj": 8,
            "AF_adj": 0.08,
            "AN_eas_adj": 16,
            "AN_sas_adj": 2,
            "AC_raw": None,
            "AF_raw": None,
            "AN_eas_raw": None,
            "AN_sas_raw": None,
        },
    ]

    # Validate each expected value.
    for row, expected in zip(rows, expected_values):
        for key, expected_value in expected.items():
            assert row[key] == expected_value, (
                f"Mismatch in row {row['idx']} for key '{key}': "
                f"expected {expected_value}, got {row[key]}"
            )


@pytest.fixture
def ht_for_check_sex_chr_metrics() -> hl.Table:
    """Fixture to set up a Hail Table with the desired structure and data for testing check_sex_chr_metrics."""
    data = [
        {
            "locus": hl.locus("chrX", 9000, reference_genome="GRCh38"),
            "info": {
                "nhomalt": 3,
                "nhomalt_XX": 2,
                "nhomalt_amr": 5,
                "nhomalt_amr_XX": 1,
                "AC": 6,
                "AC_XX": 6,
            },
        },
        {
            "locus": hl.locus("chrX", 1000000, reference_genome="GRCh38"),
            "info": {
                "nhomalt": 5,
                "nhomalt_XX": 5,
                "nhomalt_amr": 5,
                "nhomalt_amr_XX": 5,
                "AC": 10,
                "AC_XX": 10,
            },
        },
        {
            "locus": hl.locus("chrY", 1000000, reference_genome="GRCh38"),
            "info": {
                "nhomalt": 5,
                "nhomalt_XX": hl.missing(hl.tint32),
                "nhomalt_amr": hl.missing(hl.tint32),
                "nhomalt_amr_XX": hl.missing(hl.tint32),
                "AC_XX": hl.missing(hl.tint32),
                "AC": 6,
            },
        },
        {
            "locus": hl.locus("chrY", 2000000, reference_genome="GRCh38"),
            "info": {
                "nhomalt": 5,
                "nhomalt_XX": 3,
                "nhomalt_amr": hl.missing(hl.tint32),
                "nhomalt_amr_XX": hl.missing(hl.tint32),
                "AC_XX": hl.missing(hl.tint32),
                "AC": 6,
            },
        },
    ]

    ht = hl.Table.parallelize(
        data,
        hl.tstruct(
            locus=hl.tlocus(reference_genome="GRCh38"),
            info=hl.tstruct(
                nhomalt=hl.tint32,
                nhomalt_XX=hl.tint32,
                nhomalt_amr=hl.tint32,
                nhomalt_amr_XX=hl.tint32,
                AC=hl.tint32,
                AC_XX=hl.tint32,
            ),
        ),
    )
    ht = ht.key_by("locus")
    return ht


def test_check_sex_chr_metrics_logs(ht_for_check_sex_chr_metrics) -> None:
    """Test that check_sex_chr_metrics produces the expected log messages."""
    ht = ht_for_check_sex_chr_metrics
    info_metrics = [
        "nhomalt",
        "nhomalt_XX",
        "nhomalt_amr",
        "nhomalt_amr_XX",
        "AC",
        "AC_XX",
    ]
    contigs = ["chrX", "chrY"]
    verbose = False

    # Redirect logs to a buffer.
    log_stream = StringIO()
    logger = logging.getLogger("gnomad.assessment.validity_checks")
    handler = logging.StreamHandler(log_stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Run the check_sex_chr_metrics function.
    check_sex_chr_metrics(
        ht,
        info_metrics=info_metrics,
        contigs=contigs,
        verbose=verbose,
        delimiter="_",
    )

    # Capture and parse the log output.
    handler.flush()
    log_output = log_stream.getvalue()
    logger.removeHandler(handler)

    # Perform assertions on the log output.
    assert (
        "FAILED nhomalt_XX = None check for Y variants. Values found: [3]" in log_output
    )
    assert "PASSED nhomalt_amr_XX = None check for Y variants" in log_output
    assert "PASSED AC_XX = None check for Y variants" in log_output
    assert "Found 1 sites that fail nhomalt_XX == nhomalt check:" in log_output


@pytest.fixture
def ht_for_group_sums() -> hl.Table:
    """Fixture to set up a Hail Table with the desired structure and data for make_group_sum_expr_dict."""
    data = [
        {
            "idx": 0,
            "info": {
                "AC_afr_adj": 5,
                "AC_amr_adj": 10,
                "AC_adj": 15,
                "AN_afr_XX_adj": 20,
                "AN_afr_XY_adj": 30,
                "AN_adj": 50,
            },
        },
        {
            "idx": 1,
            "info": {
                "AC_afr_adj": 3,
                "AC_amr_adj": 7,
                "AC_adj": 10,
                "AN_afr_XX_adj": 15,
                "AN_afr_XY_adj": 25,
                "AN_adj": 40,
            },
        },
        {
            "idx": 2,
            "info": {
                "AC_afr_adj": 2,
                "AC_amr_adj": 3,
                "AC_adj": 5,
                "AN_afr_XX_adj": 10,
                "AN_afr_XY_adj": 20,
                "AN_adj": 35,
            },
        },
    ]

    ht = hl.Table.parallelize(
        data,
        hl.tstruct(
            idx=hl.tint32,
            info=hl.tstruct(
                AC_afr_adj=hl.tint32,
                AC_amr_adj=hl.tint32,
                AC_adj=hl.tint32,
                AN_afr_XX_adj=hl.tint32,
                AN_afr_XY_adj=hl.tint32,
                AN_adj=hl.tint32,
            ),
        ),
    )

    return ht


def test_make_group_sum_expr_dict_logs(ht_for_group_sums, caplog) -> None:
    """Test that make_group_sum_expr_dict produces the expected log messages."""
    ht = ht_for_group_sums

    subset = ""
    label_groups = {"pop": ["afr", "amr"], "group": ["adj"]}
    sort_order = ["pop", "sex"]
    delimiter = "_"
    metric_first_field = True
    metrics = ["AC", "AN"]

    with caplog.at_level(logging.INFO, logger="gnomad.assessment.validity_checks"):
        make_group_sum_expr_dict(
            ht, subset, label_groups, sort_order, delimiter, metric_first_field, metrics
        )
    log_messages = [record.getMessage().lower().strip() for record in caplog.records]

    # Perform assertions on log output (does not include all expected log messages).
    expected_logs = [
        "including field ac_afr_adj",
        "including field ac_amr_adj",
        "an_afr_adj is not in table's info field, it will not be included in make_group_sum_expr_dict",
        "an_amr_adj is not in table's info field, it will not be included in make_group_sum_expr_dict",
        "generated annot_dict keys: ['sum_ac_adj_pop', 'sum_an_adj_pop']",
        "no valid fields found for sum_an_adj_pop",
    ]

    for log_phrase in expected_logs:
        assert any(
            log_phrase in log for log in log_messages
        ), f"Expected phrase missing: {log_phrase}"


def test_sum_group_callstats(ht_for_group_sums, caplog) -> None:
    """Test that sum_group_callstats produces the expected log messages."""
    ht = ht_for_group_sums

    sexes = ["XX", "XY"]
    subsets = [""]
    pops = ["afr", "amr"]
    groups = ["adj"]
    metrics = ["AC", "AN"]

    with caplog.at_level(logging.INFO, logger="gnomad.assessment.validity_checks"):
        sum_group_callstats(
            ht,
            sexes=sexes,
            subsets=subsets,
            pops=pops,
            groups=groups,
            metrics=metrics,
            verbose=True,
            delimiter="_",
            gen_anc_label_name="gen_anc",
        )

    # Convert expected log messages to lowercase and strip whitespace
    log_messages = [record.getMessage().lower().strip() for record in caplog.records]

    expected_logs = [
        "passed ac_adj = sum_ac_adj_gen_anc check",
        "found 3 sites that fail an_adj = sum_an_adj_gen_anc check",
        "found 3 sites that fail ac_adj = sum_ac_adj_sex check",
        "found 3 sites that fail an_adj = sum_an_adj_sex check",
        "found 3 sites that fail ac_adj = sum_ac_adj_gen_anc_sex check",
        "found 1 sites that fail an_adj = sum_an_adj_gen_anc_sex check",
    ]

    for log_phrase in expected_logs:
        assert any(
            log_phrase in log for log in log_messages
        ), f"Expected phrase missing: {log_phrase}"


@pytest.fixture
def ht_for_check_global_and_row_annot_lengths() -> hl.Table:
    """Fixture to set up a Hail Table with the desired structure and data for check_global_and_row_annot_lengths."""
    ht = hl.Table.parallelize(
        [
            {"freq": [0.1, 0.2, 0.3], "faf": [0.01, 0.02]},
            {"freq": [0.8, 0.4, 0.5], "faf": [0.03, 0.04, 0.05]},
        ],
        hl.tstruct(freq=hl.tarray(hl.tfloat64), faf=hl.tarray(hl.tfloat64)),
    )

    return ht.annotate_globals(
        freq_meta=["A", "B", "C"],
        freq_index_dict={"A": 0, "B": 1, "C": 2},
        freq_meta_sample_count=[100, 200, 300],
        faf_meta=["D", "E"],
        faf_index_dict={"D": 0, "E": 1},
    )


def test_check_global_and_row_annot_lengths(
    ht_for_check_global_and_row_annot_lengths, caplog
) -> None:
    """Test that check_global_and_row_annot_lengths produces the expected log messages."""
    ht = ht_for_check_global_and_row_annot_lengths

    # Define the row_to_globals_check dictionary.
    row_to_globals_check = {
        "freq": ["freq_meta", "freq_index_dict", "freq_meta_sample_count"],
        "faf": ["faf_meta", "faf_index_dict"],
    }

    with caplog.at_level(logging.INFO, logger="gnomad.assessment.validity_checks"):
        check_global_and_row_annot_lengths(
            ht, row_to_globals_check, check_all_rows=True
        )

    log_messages = [record.message for record in caplog.records]

    # Verify log messages.
    expected_logs = [
        "Passed global and row lengths comparison: Length of freq_meta in globals (3) does match length of freq in 2 out of 2 rows (row length counter: {3: 2})",
        "Passed global and row lengths comparison: Length of freq_index_dict in globals (3) does match length of freq in 2 out of 2 rows (row length counter: {3: 2})",
        "Passed global and row lengths comparison: Length of freq_meta_sample_count in globals (3) does match length of freq in 2 out of 2 rows (row length counter: {3: 2})",
        "Failed global and row lengths comparison: Length of faf_meta in globals (2) does NOT match length of faf in 1 out of 2 rows (row length counter: {2: 1, 3: 1})",
        "Failed global and row lengths comparison: Length of faf_index_dict in globals (2) does NOT match length of faf in 1 out of 2 rows (row length counter: {2: 1, 3: 1})",
    ]

    for msg in expected_logs:
        assert msg in log_messages, f"Expected log message is missing: {msg}"


@pytest.fixture
def ht_for_check_raw_and_adj_callstats() -> hl.Table:
    """Fixture to create a Hail Table with the expected structure and test values for check_raw_and_adj_callstats, using underscore as the delimiter."""
    data = [
        {
            "idx": 0,
            "info": {
                "AC_raw": 5,
                "AC_adj": 3,
                "AF_raw": 0.02,
                "AF_adj": 0.01,
                "AN_raw": 2500,
                "AN_adj": 2400,
                "nhomalt_raw": 1,  # Defined since AN_raw is defined
                "nhomalt_adj": 0,
            },
            "filters": hl.empty_set(hl.tstr),
        },
        {
            "idx": 1,
            "info": {
                "AC_raw": 0,
                "AC_adj": 0,
                "AF_raw": 0.0,
                "AF_adj": 0.0,
                "AN_raw": 0,
                "AN_adj": 0,
                "nhomalt_raw": None,
                "nhomalt_adj": None,
            },
            "filters": hl.empty_set(hl.tstr),
        },
        {
            "idx": 2,
            "info": {
                "AC_raw": -1,
                "AC_adj": -1,
                "AF_raw": -0.01,
                "AF_adj": -0.01,
                "AN_raw": 1000,
                "AN_adj": 1100,
                "nhomalt_raw": -3,
                "nhomalt_adj": 2,
            },
            "filters": {"LowQual"},
        },
        {
            "idx": 3,
            "info": {
                "AC_raw": 10,
                "AF_raw": 0.05,
                "AN_raw": 3000,
                "AN_adj": 2000,
                "AC_adj": 8,
                "AF_adj": 0.02,
                "nhomalt_raw": 3,
                "nhomalt_adj": 1,
            },
            "filters": hl.empty_set(hl.tstr),
        },
        {
            "idx": 4,
            "info": {
                "AC_raw": None,
                "AF_raw": 0.05,
                "AN_raw": None,
                "AN_adj": 1000,
                "AC_adj": 8,
                "AF_adj": 0.03,
                "nhomalt_raw": None,
                "nhomalt_adj": 1,
            },
            "filters": hl.empty_set(hl.tstr),
        },
    ]

    ht = hl.Table.parallelize(
        data,
        hl.tstruct(
            idx=hl.tint32,
            info=hl.tstruct(
                AC_raw=hl.tint32,
                AC_adj=hl.tint32,
                AF_raw=hl.tfloat64,
                AF_adj=hl.tfloat64,
                AN_raw=hl.tint32,
                AN_adj=hl.tint32,
                nhomalt_raw=hl.tint32,
                nhomalt_adj=hl.tint32,
            ),
            filters=hl.tset(hl.tstr),
        ),
    )

    return ht


def test_check_raw_and_adj_callstats(
    ht_for_check_raw_and_adj_callstats, caplog
) -> None:
    """Test check_raw_and_adj_callstats function and it's expected log output."""
    ht = ht_for_check_raw_and_adj_callstats
    with caplog.at_level(logging.INFO, logger="gnomad.assessment.validity_checks"):
        check_raw_and_adj_callstats(
            ht, subsets=[""], verbose=True, delimiter="_", metric_first_field=True
        )

    log_messages = [record.getMessage() for record in caplog.records]

    expected_logs = [
        # Expected PASSES.
        "PASSED AC_raw defined when AN defined and missing when AN missing check",
        "PASSED AC_adj defined when AN defined and missing when AN missing check",
        "PASSED AF_adj defined when AN defined (and > 0) and missing when AN missing check",
        "PASSED nhomalt_raw <= AC_raw / 2 check",
        # Expected FAILURES.
        "Found 1 sites that fail nhomalt_raw defined when AN defined and missing when AN missing check:",
        "Found 1 sites that fail AF_raw defined when AN defined (and > 0) and missing when AN missing check:",
        "Found 1 sites that fail AF_raw missing when AN 0 check:",
        "Found 1 sites that fail nhomalt_adj defined when AN defined and missing when AN missing check:",
        "Found 1 sites that fail AF_adj missing when AN 0 check:",
        "Found 2 sites that fail AC_raw > 0 check:",
        "Found 1 sites that fail AC_adj >= 0 check:",
        "Found 1 sites that fail AC_raw >= AC_adj check",
        "Found 2 sites that fail AF_raw > 0 check:",
        "Found 1 sites that fail AF_adj >= 0 check:",
        "Found 2 sites that fail AN_raw >= AN_adj check:",
        "Found 2 sites that fail nhomalt_raw >= nhomalt_adj check:",
        "Found 1 sites that fail nhomalt_adj <= AC_adj / 2 check:",
    ]

    for log_phrase in expected_logs:
        assert any(
            log_phrase in log for log in log_messages
        ), f"Expected phrase missing: {log_phrase}"

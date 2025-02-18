"""Tests for the filtering module."""

from typing import Any, Dict, List, Union

import hail as hl
import pytest

from gnomad.utils.filtering import filter_arrays_by_meta, filter_meta_array


@pytest.fixture(scope="class")
def metadata_combinations():
    """Top-level fixture to hold all metadata combinations."""

    class MetaDataCombinations:
        only_group = [{"group": "adj"}, {"group": "raw"}]
        group_gen_anc_a = [{"group": "adj", "gen_anc": "a"}]
        group_gen_anc_a_b = [*group_gen_anc_a, {"group": "adj", "gen_anc": "b"}]
        group_gen_anc = [*group_gen_anc_a_b, {"group": "adj", "gen_anc": "c"}]
        group_sex = [{"group": "adj", "sex": "XX"}, {"group": "adj", "sex": "XY"}]
        group_subset = [
            {"group": "adj", "subset": "s1"},
            {"group": "raw", "subset": "s1"},
        ]
        group_gen_anc_a_sex = [
            {"group": "adj", "gen_anc": "a", "sex": "XX"},
            {"group": "adj", "gen_anc": "a", "sex": "XY"},
        ]
        group_gen_anc_b_sex = [
            {"group": "adj", "gen_anc": "b", "sex": "XX"},
            {"group": "adj", "gen_anc": "b", "sex": "XY"},
        ]
        group_gen_anc_a_b_sex = group_gen_anc_a_sex + group_gen_anc_b_sex
        group_gen_anc_sex = [
            *group_gen_anc_a_b_sex,
            {"group": "adj", "gen_anc": "c", "sex": "XX"},
            {"group": "adj", "gen_anc": "c", "sex": "XY"},
        ]
        group_gen_anc_a_subset = [{"group": "adj", "gen_anc": "a", "subset": "s1"}]
        group_gen_anc_a_b_subset = [
            *group_gen_anc_a_subset,
            {"group": "adj", "gen_anc": "b", "subset": "s1"},
        ]
        group_gen_anc_subset = [
            *group_gen_anc_a_b_subset,
            {"group": "adj", "gen_anc": "c", "subset": "s1"},
        ]
        group_sex_subset = [
            {"group": "adj", "sex": "XX", "subset": "s1"},
            {"group": "adj", "sex": "XY", "subset": "s1"},
        ]
        group_gen_anc_a_sex_subset = [
            {"group": "adj", "gen_anc": "a", "sex": "XX", "subset": "s1"},
            {"group": "adj", "gen_anc": "a", "sex": "XY", "subset": "s1"},
        ]
        group_gen_anc_b_sex_subset = [
            {"group": "adj", "gen_anc": "b", "sex": "XX", "subset": "s1"},
            {"group": "adj", "gen_anc": "b", "sex": "XY", "subset": "s1"},
        ]
        group_gen_anc_a_b_sex_subset = (
            group_gen_anc_a_sex_subset + group_gen_anc_b_sex_subset
        )
        group_gen_anc_sex_subset = [
            *group_gen_anc_a_b_sex_subset,
            {"group": "adj", "gen_anc": "c", "sex": "XX", "subset": "s1"},
            {"group": "adj", "gen_anc": "c", "sex": "XY", "subset": "s1"},
        ]
        group_gen_anc_a_downsampling = [
            {"group": "adj", "gen_anc": "a", "downsampling": "1"},
            {"group": "adj", "gen_anc": "a", "downsampling": "2"},
            {"group": "adj", "gen_anc": "a", "downsampling": "3"},
        ]
        group_gen_anc_a_b_downsampling = [
            *group_gen_anc_a_downsampling,
            {"group": "adj", "gen_anc": "b", "downsampling": "1"},
            {"group": "adj", "gen_anc": "b", "downsampling": "2"},
            {"group": "adj", "gen_anc": "b", "downsampling": "3"},
        ]
        group_gen_anc_downsampling = [
            *group_gen_anc_a_b_downsampling,
            {"group": "adj", "gen_anc": "c", "downsampling": "1"},
            {"group": "adj", "gen_anc": "c", "downsampling": "2"},
            {"group": "adj", "gen_anc": "c", "downsampling": "3"},
        ]
        group_gen_anc_a_subset_downsampling = [
            {"group": "adj", "gen_anc": "a", "subset": "s1", "downsampling": "1"},
            {"group": "adj", "gen_anc": "a", "subset": "s1", "downsampling": "2"},
            {"group": "adj", "gen_anc": "a", "subset": "s1", "downsampling": "3"},
        ]
        group_gen_anc_a_b_subset_downsampling = [
            *group_gen_anc_a_subset_downsampling,
            {"group": "adj", "gen_anc": "b", "subset": "s1", "downsampling": "1"},
            {"group": "adj", "gen_anc": "b", "subset": "s1", "downsampling": "2"},
            {"group": "adj", "gen_anc": "b", "subset": "s1", "downsampling": "3"},
        ]
        group_gen_anc_subset_downsampling = [
            *group_gen_anc_a_b_subset_downsampling,
            {"group": "adj", "gen_anc": "c", "subset": "s1", "downsampling": "1"},
            {"group": "adj", "gen_anc": "c", "subset": "s1", "downsampling": "2"},
            {"group": "adj", "gen_anc": "c", "subset": "s1", "downsampling": "3"},
        ]
        downsampling = group_gen_anc_downsampling + group_gen_anc_subset_downsampling
        sex = (
            group_sex + group_gen_anc_sex + group_sex_subset + group_gen_anc_sex_subset
        )
        no_sex = (
            only_group
            + group_gen_anc
            + group_subset
            + group_gen_anc_subset
            + group_gen_anc_downsampling
            + group_gen_anc_subset_downsampling
        )
        sex_and_gen_anc = group_gen_anc_sex + group_gen_anc_sex_subset
        no_sex_and_no_gen_anc = only_group + group_subset
        sex_or_subset = (
            group_sex
            + group_subset
            + group_gen_anc_sex
            + group_gen_anc_subset
            + group_sex_subset
            + group_gen_anc_sex_subset
            + group_gen_anc_subset_downsampling
        )
        sex_and_gen_anc_a = group_gen_anc_a_sex + group_gen_anc_a_sex_subset
        sex_or_gen_anc_a = (
            group_gen_anc_a
            + group_sex
            + group_gen_anc_sex
            + group_gen_anc_a_subset
            + group_sex_subset
            + group_gen_anc_sex_subset
            + group_gen_anc_a_downsampling
            + group_gen_anc_a_subset_downsampling
        )
        sex_and_gen_anc_a_or_b = (
            group_gen_anc_a_sex
            + group_gen_anc_b_sex
            + group_gen_anc_a_sex_subset
            + group_gen_anc_b_sex_subset
        )
        no_downsampling = (
            only_group
            + group_gen_anc
            + group_sex
            + group_subset
            + group_gen_anc_sex
            + group_gen_anc_subset
            + group_sex_subset
            + group_gen_anc_sex_subset
        )
        no_subset_and_no_downsampling = (
            only_group + group_gen_anc + group_sex + group_gen_anc_sex
        )
        no_subset_or_no_downsampling = no_downsampling + group_gen_anc_downsampling
        no_downsampling_and_no_gen_anc_c = (
            only_group
            + group_gen_anc_a_b
            + group_sex
            + group_subset
            + group_gen_anc_a_b_sex
            + group_gen_anc_a_b_subset
            + group_sex_subset
            + group_gen_anc_a_b_sex_subset
        )
        no_downsampling_or_no_gen_anc_c = (
            no_downsampling
            + group_gen_anc_a_b_downsampling
            + group_gen_anc_a_b_subset_downsampling
        )
        sex_and_no_subset = group_sex + group_gen_anc_sex
        sex_or_no_subset = (
            only_group
            + group_gen_anc
            + group_sex
            + group_gen_anc_sex
            + group_sex_subset
            + group_gen_anc_sex_subset
            + group_gen_anc_downsampling
        )

    return MetaDataCombinations


@pytest.fixture
def mock_meta_expr(metadata_combinations) -> hl.expr.ArrayExpression:
    """Mock meta expression."""
    return hl.literal(
        metadata_combinations.only_group
        + metadata_combinations.group_gen_anc
        + metadata_combinations.group_sex
        + metadata_combinations.group_subset
        + metadata_combinations.group_gen_anc_sex
        + metadata_combinations.group_gen_anc_subset
        + metadata_combinations.group_sex_subset
        + metadata_combinations.group_gen_anc_sex_subset
        + metadata_combinations.group_gen_anc_downsampling
        + metadata_combinations.group_gen_anc_subset_downsampling
    )


class TestFilterMetaArray:
    """Tests for the filter_meta_array function."""

    # Define some common parameters.
    all_and = ["and", "and", "and"]
    s_ga_list = ["sex", "gen_anc"]
    s_ss_list = ["sex", "subset"]
    s_list = ["sex"]
    g_s_list = ["group", "sex"]
    ss_d_ex = [None, ["subset", "downsampling"], None, None]
    ds_ex = [None, ["downsampling"], None]
    s_ss = [["sex"], ["subset"]]

    ga_a = {"gen_anc": "a"}
    ga_c = {"gen_anc": "c"}
    ga_ab = {"gen_anc": ["a", "b"]}

    @pytest.mark.parametrize(
        "keys_to_keep, keys_to_exclude, key_value_pairs_to_keep, key_value_pairs_to_exclude, keep_combine_operator, exclude_combine_operator, combine_operator, exact_match, expected",
        [
            (s_list, None, None, None, *all_and, False, "sex"),
            (s_ga_list, None, None, None, *all_and, False, "sex_and_gen_anc"),
            (g_s_list, None, None, None, *all_and, True, "group_sex"),
            (s_ss_list, None, None, None, "or", "and", "and", False, "sex_or_subset"),
            (s_list, None, ga_a, None, *all_and, False, "sex_and_gen_anc_a"),
            (s_list, None, ga_a, None, "or", "and", "and", False, "sex_or_gen_anc_a"),
            (g_s_list, None, ga_a, None, *all_and, True, "group_gen_anc_a_sex"),
            (s_list, None, ga_ab, None, *all_and, False, "sex_and_gen_anc_a_or_b"),
            (*ds_ex, None, *all_and, False, "no_downsampling"),
            (*ss_d_ex, *all_and, False, "no_subset_and_no_downsampling"),
            (*ss_d_ex, "and", "or", "and", False, "no_subset_or_no_downsampling"),
            (*ds_ex, ga_c, *all_and, False, "no_downsampling_and_no_gen_anc_c"),
            (
                *ds_ex,
                ga_c,
                "and",
                "or",
                "and",
                False,
                "no_downsampling_or_no_gen_anc_c",
            ),
            (*s_ss, None, None, *all_and, False, "sex_and_no_subset"),
            (*s_ss, None, None, "and", "and", "or", False, "sex_or_no_subset"),
        ],
    )
    def test_filter_meta_array(
        self,
        mock_meta_expr: hl.expr.ArrayExpression,
        keys_to_keep: List[str],
        keys_to_exclude: List[str],
        key_value_pairs_to_keep: Dict[str, Any],
        key_value_pairs_to_exclude: Dict[str, Any],
        keep_combine_operator: str,
        exclude_combine_operator: str,
        combine_operator: str,
        exact_match: bool,
        expected: str,
        metadata_combinations: Any,
    ) -> None:
        """Test filter_meta_array function."""
        result = filter_meta_array(
            meta_expr=mock_meta_expr,
            keys_to_keep=keys_to_keep,
            keys_to_exclude=keys_to_exclude,
            key_value_pairs_to_keep=key_value_pairs_to_keep,
            key_value_pairs_to_exclude=key_value_pairs_to_exclude,
            keep_combine_operator=keep_combine_operator,
            exclude_combine_operator=exclude_combine_operator,
            combine_operator=combine_operator,
            exact_match=exact_match,
        )
        assert hl.eval(result) == getattr(metadata_combinations, expected)


class TestFilterArraysByMeta:
    """Tests for the filter_arrays_by_meta function."""

    @pytest.fixture
    def simple_mock_meta_expr(self):
        """Get simple mock meta expression for filter_arrays_by_meta."""
        return hl.literal(
            [
                {"key1": "value1", "key2": "value2"},
                {"key1": "value3", "key2": "value4"},
                {"key1": "value5", "key2": "value6"},
            ]
        )

    @pytest.fixture
    def simple_mock_meta_indexed_exprs(self):
        """Get simple mock meta-indexed expressions for filter_arrays_by_meta."""
        return {
            "expr1": hl.literal([1, 2, 3]),
            "expr2": hl.literal([4, 5, 6]),
        }

    params = {
        "k1_keep": {
            "in": {"key1": {"values": ["value1", "value3"], "keep": True}},
            "out": (
                [
                    {"key1": "value1", "key2": "value2"},
                    {"key1": "value3", "key2": "value4"},
                ],
                [1, 2],
                [4, 5],
            ),
        },
        "k1_ex": {
            "in": {"key1": {"values": ["value1", "value3"], "keep": False}},
            "out": ([{"key1": "value5", "key2": "value6"}], [3], [6]),
        },
        "k12_keep": {
            "in": {
                "key1": {"values": ["value1"], "keep": True},
                "key2": {"values": ["value2"], "keep": True},
            },
            "out": ([{"key1": "value1", "key2": "value2"}], [1], [4]),
        },
    }

    @pytest.mark.parametrize(
        "items_to_filter, keep, combine_operator, exact_match, expected_meta, expected_expr1, expected_expr2",
        [
            (params["k1_keep"]["in"], True, "and", False, *params["k1_keep"]["out"]),
            (params["k1_ex"]["in"], False, "and", False, *params["k1_ex"]["out"]),
            (params["k12_keep"]["in"], True, "and", True, *params["k12_keep"]["out"]),
        ],
    )
    def test_filter_arrays_by_meta(
        self,
        simple_mock_meta_expr: hl.expr.ArrayExpression,
        simple_mock_meta_indexed_exprs: Dict[str, hl.expr.ArrayExpression],
        items_to_filter: Dict[str, Dict[str, Any]],
        keep: bool,
        combine_operator: str,
        exact_match: bool,
        expected_meta: List[Dict[str, str]],
        expected_expr1: List[int],
        expected_expr2: List[int],
    ) -> None:
        """Test filter_arrays_by_meta function."""
        filtered_meta_expr, filtered_meta_indexed_exprs = filter_arrays_by_meta(
            meta_expr=simple_mock_meta_expr,
            meta_indexed_exprs=simple_mock_meta_indexed_exprs,
            items_to_filter=items_to_filter,
            keep=keep,
            combine_operator=combine_operator,
            exact_match=exact_match,
        )
        assert hl.eval(filtered_meta_expr) == expected_meta
        assert hl.eval(filtered_meta_indexed_exprs["expr1"]) == expected_expr1
        assert hl.eval(filtered_meta_indexed_exprs["expr2"]) == expected_expr2

    # Additional cases reusing complex cases from the metadata_combinations fixture.
    all_and = ["and", "and", "and"]
    s_ga_list = ["sex", "gen_anc"]
    ss_d_list = ["subset", "downsampling"]
    s_ga_dict = {"sex": None, "gen_anc": None}
    s_ga_keep = {"sex": {"keep": True}, "gen_anc": {"keep": True}}
    s_ga_ex = {"sex": {"keep": False}, "gen_anc": {"keep": False}}
    s_ga_a = {"sex": None, "gen_anc": "a"}
    s_ga_a_2 = {"sex": None, "gen_anc": ["a"]}
    s_ga_a_3 = {"sex": None, "gen_anc": {"values": "a"}}
    s_ga_a_4 = {"sex": None, "gen_anc": {"values": ["a"]}}
    s_ga_a_4_keep = {"sex": None, "gen_anc": {"values": ["a"], "keep": True}}
    g_s_ga_a = {"group": None, "sex": None, "gen_anc": "a"}
    s_ga_a_b = {"sex": None, "gen_anc": ["a", "b"]}
    d_ga_c = {"downsampling": None, "gen_anc": "c"}
    s_keep_ss_ex = {"sex": {"keep": True}, "subset": {"keep": False}}

    @pytest.mark.parametrize(
        "items_to_filter, keep, keep_combine_operator, exclude_combine_operator, combine_operator, exact_match, expected_meta",
        [
            (["sex"], True, *all_and, False, "sex"),
            ({"sex": None}, True, *all_and, False, "sex"),
            ({"sex": {"keep": True}}, True, *all_and, False, "sex"),
            ({"sex": {"keep": True}}, False, *all_and, False, "sex"),
            (["sex"], True, "or", "or", "or", False, "sex"),
            (["sex"], False, *all_and, False, "no_sex"),
            ({"sex": None}, False, *all_and, False, "no_sex"),
            ({"sex": {"keep": False}}, True, *all_and, False, "no_sex"),
            (s_ga_list, True, *all_and, False, "sex_and_gen_anc"),
            (s_ga_dict, True, *all_and, False, "sex_and_gen_anc"),
            (s_ga_keep, True, *all_and, False, "sex_and_gen_anc"),
            (s_ga_list, False, *all_and, False, "no_sex_and_no_gen_anc"),
            (s_ga_ex, True, *all_and, False, "no_sex_and_no_gen_anc"),
            (["sex", "subset"], True, "or", "and", "and", False, "sex_or_subset"),
            (ss_d_list, False, *all_and, False, "no_subset_and_no_downsampling"),
            (
                ss_d_list,
                False,
                "and",
                "or",
                "and",
                False,
                "no_subset_or_no_downsampling",
            ),
            (["group", "sex"], True, *all_and, True, "group_sex"),
            (["group"], True, *all_and, True, "only_group"),
            (s_ga_a, True, *all_and, False, "sex_and_gen_anc_a"),
            (s_ga_a_2, True, *all_and, False, "sex_and_gen_anc_a"),
            (s_ga_a_3, True, *all_and, False, "sex_and_gen_anc_a"),
            (s_ga_a_4, True, *all_and, False, "sex_and_gen_anc_a"),
            (s_ga_a_4_keep, True, *all_and, False, "sex_and_gen_anc_a"),
            (s_ga_a, True, "or", "and", "and", False, "sex_or_gen_anc_a"),
            (g_s_ga_a, True, *all_and, True, "group_gen_anc_a_sex"),
            (s_ga_a_b, True, *all_and, False, "sex_and_gen_anc_a_or_b"),
            (
                d_ga_c,
                False,
                "and",
                "or",
                "and",
                False,
                "no_downsampling_or_no_gen_anc_c",
            ),
            (s_keep_ss_ex, True, *all_and, False, "sex_and_no_subset"),
            (s_keep_ss_ex, True, "and", "and", "or", False, "sex_or_no_subset"),
        ],
    )
    def test_filter_arrays_by_meta_with_reuse(
        self,
        mock_meta_expr: hl.expr.ArrayExpression,
        items_to_filter: Union[List[str], Dict[str, Dict[str, Any]]],
        keep: bool,
        keep_combine_operator,
        exclude_combine_operator,
        combine_operator: str,
        exact_match: bool,
        expected_meta: str,
        metadata_combinations: Any,
    ) -> None:
        """Test filter_arrays_by_meta function with reused cases."""
        filtered_meta_expr, filtered_meta_indexed_exprs = filter_arrays_by_meta(
            meta_expr=mock_meta_expr,
            meta_indexed_exprs={"meta_array": mock_meta_expr},
            items_to_filter=items_to_filter,
            keep=keep,
            keep_combine_operator=keep_combine_operator,
            exclude_combine_operator=exclude_combine_operator,
            combine_operator=combine_operator,
            exact_match=exact_match,
        )
        assert hl.eval(filtered_meta_expr) == getattr(
            metadata_combinations, expected_meta
        )
        assert hl.eval(filtered_meta_indexed_exprs["meta_array"]) == getattr(
            metadata_combinations, expected_meta
        )

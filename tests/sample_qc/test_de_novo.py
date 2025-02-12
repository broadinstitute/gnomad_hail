"""Test suite for de novo mutation functions."""

import hail as hl
import pytest

from gnomad.sample_qc.relatedness import (
    calculate_de_novo_post_prob,
    default_get_de_novo_expr,
)


class TestDeNovoMutation:
    """Test suite for de novo mutation functions."""

    @pytest.mark.parametrize(
        "proband_pl, father_pl, mother_pl, diploid, hemi_x, hemi_y, freq_prior, min_pop_prior, expected",
        [
            # Valid test cases (should return expected numeric values)
            (
                [73, 0, 161],
                [0, 99, 198],
                [0, 99, 198],
                True,
                False,
                False,
                1.80e-02,
                100 / 3e7,
                0.999,
            ),
            (
                [152, 0, 283],
                [0, 99, 198],
                [0, 63, 126],
                True,
                False,
                False,
                7.55e-02,
                100 / 3e7,
                0.198,
            ),
            # Invalid `freq_prior` case (should raise `HailUserError`)
            (
                [99, 50, 0],
                [0, 99, 198],
                [0, 99, 198],
                False,
                True,
                False,
                1.2,
                100 / 3e7,
                None,
            ),
        ],
    )
    def test_calculate_de_novo_post_prob(
        self,
        proband_pl,
        father_pl,
        mother_pl,
        diploid,
        hemi_x,
        hemi_y,
        freq_prior,
        min_pop_prior,
        expected,
    ):
        """Test `calculate_de_novo_post_prob` function."""
        # Case where we expect an error (`freq_prior` is out of range)
        if expected is None:
            with pytest.raises(
                hl.utils.HailUserError,
                match=r"de_novo: expect 0 <= freq_prior_expr <= 1, found .*",
            ):
                hl.eval(
                    calculate_de_novo_post_prob(
                        hl.literal(proband_pl),
                        hl.literal(father_pl),
                        hl.literal(mother_pl),
                        hl.literal(diploid),
                        hl.literal(hemi_x),
                        hl.literal(hemi_y),
                        hl.literal(freq_prior),
                        min_pop_prior,
                    )
                )
        else:
            # Case where we expect a valid float result
            p_dn_expr = calculate_de_novo_post_prob(
                hl.literal(proband_pl),
                hl.literal(father_pl),
                hl.literal(mother_pl),
                hl.literal(diploid),
                hl.literal(hemi_x),
                hl.literal(hemi_y),
                hl.literal(freq_prior),
                min_pop_prior,
            )

            # Assert with floating-point tolerance
            assert round(hl.eval(p_dn_expr), 3) == expected

    @pytest.mark.parametrize(
        "locus, alleles, proband_expr, father_expr, mother_expr, is_xx_expr, freq_prior_expr, expected_exception, expected_result",
        [
            # 1. Autosomal locus with HIGH confidence
            (
                hl.locus("chr1", 10000, reference_genome="GRCh38"),
                hl.literal(["A", "C"]),
                hl.struct(GT=hl.call(0, 1), AD=[5, 30], DP=35, GQ=99, PL=[99, 0, 1]),
                hl.struct(GT=hl.call(0, 0), AD=[20, 0], DP=20, GQ=60, PL=[0, 60, 120]),
                hl.struct(GT=hl.call(0, 0), AD=[25, 0], DP=25, GQ=80, PL=[0, 80, 150]),
                hl.literal(True),
                hl.literal(1e-5),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=0.999,
                    confidence="HIGH",
                    fail_reason=hl.missing(hl.tset(hl.tstr)),
                ),
            ),
            # 2. Autosomal locus with MEDIUM confidence
            (
                hl.locus("chr1", 11000, reference_genome="GRCh38"),
                hl.literal(["CT", "C"]),
                hl.struct(
                    GT=hl.call(0, 1), AD=[59, 61], DP=120, GQ=99, PL=[542, 0, 1940]
                ),
                hl.struct(GT=hl.call(0, 0), AD=[32, 0], DP=32, GQ=60, PL=[0, 60, 120]),
                hl.struct(GT=hl.call(0, 0), AD=[37, 0], DP=37, GQ=60, PL=[0, 60, 120]),
                hl.literal(False),
                hl.literal(2.62e-03),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=0.615,
                    confidence="MEDIUM",
                    fail_reason=hl.missing(hl.tset(hl.tstr)),
                ),
            ),
            # 3. Autosomal locus with LOW confidence
            (
                hl.locus("chr1", 12000, reference_genome="GRCh38"),
                hl.literal(["G", "T"]),
                hl.struct(GT=hl.call(0, 1), AD=[7, 2], DP=18, GQ=43, PL=[43, 0, 387]),
                hl.struct(GT=hl.call(0, 0), AD=[25, 0], DP=25, GQ=40, PL=[0, 40, 80]),
                hl.struct(GT=hl.call(0, 0), AD=[23, 0], DP=23, GQ=40, PL=[0, 40, 80]),
                hl.literal(True),
                hl.literal(0),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=0.926,
                    confidence="LOW",
                    fail_reason=hl.missing(hl.tset(hl.tstr)),
                ),
            ),
            # 4. Autosomal locus with one FAIL condition
            (
                hl.locus("chr1", 13000, reference_genome="GRCh38"),
                hl.literal(["T", "TA"]),
                hl.struct(GT=hl.call(0, 1), AD=[4, 2], DP=8, GQ=30, PL=[30, 0, 103]),
                hl.struct(GT=hl.call(0, 0), AD=[7, 0], DP=7, GQ=24, PL=[0, 24, 48]),
                hl.struct(GT=hl.call(0, 0), AD=[2, 0], DP=2, GQ=6, PL=[0, 6, 12]),
                hl.literal(True),
                hl.literal(2.00e-01),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=hl.missing(hl.tfloat64),
                    confidence=hl.missing(hl.tstr),
                    fail_reason={"min_de_novo_p"},
                ),
            ),
            # 5. Autosomal locus with multiple FAIL conditions
            (
                hl.locus("chr1", 14000, reference_genome="GRCh38"),
                hl.literal(["A", "G"]),
                hl.struct(GT=hl.call(0, 1), AD=[9, 2], DP=11, GQ=2, PL=[2, 0, 230]),
                hl.struct(GT=hl.call(0, 0), AD=[0, 0], DP=0, GQ=0, PL=[0, 0, 0]),
                hl.struct(GT=hl.call(0, 0), AD=[0, 0], DP=0, GQ=0, PL=[0, 0, 0]),
                hl.literal(True),
                hl.literal(1e-5),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=hl.missing(hl.tfloat64),
                    confidence=hl.missing(hl.tstr),
                    fail_reason={
                        "min_de_novo_p",
                        "min_proband_ab",
                        "min_proband_gq",
                        "parent_sum_ad_0",
                    },
                ),
            ),
            # 6. Hemi X locus for XY individual with HIGH confidence
            (
                hl.locus("chrX", 8400000, reference_genome="GRCh38"),
                hl.literal(["A", "G"]),
                hl.struct(GT=hl.call(1, 1), AD=[0, 14], DP=14, GQ=42, PL=[419, 42, 0]),
                hl.struct(GT=hl.call(0, 0), AD=[38, 0], DP=38, GQ=40, PL=[0, 40, 80]),
                hl.struct(GT=hl.call(0, 0), AD=[97, 0], DP=110, GQ=99, PL=[0, 99, 198]),
                hl.literal(False),
                hl.literal(3.74e-02),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=0.999,
                    confidence="HIGH",
                    fail_reason=hl.missing(hl.tset(hl.tstr)),
                ),
            ),
            # 7. Hemi Y locus for XY individual with HIGH confidence
            (
                hl.locus("chrY", 9900000, reference_genome="GRCh38"),
                hl.literal(["A", "G"]),
                hl.struct(
                    GT=hl.call(1, 1), AD=[0, 43], DP=43, GQ=99, PL=[1363, 129, 0]
                ),
                hl.struct(GT=hl.call(0, 0), AD=[28, 0], DP=28, GQ=40, PL=[0, 40, 80]),
                hl.struct(GT=hl.call(0, 0), AD=[0, 0], DP=0, GQ=0, PL=[0, 0, 0]),
                hl.literal(False),
                hl.missing(hl.tfloat64),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=0.962,
                    confidence="HIGH",
                    fail_reason=hl.missing(hl.tset(hl.tstr)),
                ),
            ),
            # 8. Autosomal locus that is not de novo
            (
                hl.locus("chr1", 15000, reference_genome="GRCh38"),
                hl.literal(["G", "T"]),
                hl.struct(GT=hl.call(0, 1), AD=[15, 10], DP=30, GQ=50, PL=[10, 0, 100]),
                hl.struct(GT=hl.call(0, 1), AD=[10, 5], DP=20, GQ=40, PL=[0, 20, 80]),
                hl.struct(GT=hl.call(0, 0), AD=[20, 0], DP=20, GQ=50, PL=[0, 99, 198]),
                hl.literal(False),
                hl.literal(1e-5),
                False,
                hl.struct(
                    is_de_novo=False,
                    p_de_novo=hl.missing(hl.tfloat64),
                    confidence=hl.missing(hl.tstr),
                    fail_reason=hl.missing(hl.tset(hl.tstr)),
                ),
            ),
            # 9. Autosomal locus with PLs all [0,0,0] and no freq prior
            (
                hl.locus("chr1", 16000, reference_genome="GRCh38"),
                hl.literal(["G", "T"]),
                hl.struct(GT=hl.call(0, 1), AD=[0, 2], DP=2, GQ=0, PL=[0, 0, 0]),
                hl.struct(GT=hl.call(0, 0), AD=[2, 0], DP=2, GQ=0, PL=[0, 0, 0]),
                hl.struct(GT=hl.call(0, 0), AD=[2, 0], DP=2, GQ=0, PL=[0, 0, 0]),
                hl.literal(False),
                hl.missing(hl.tfloat64),
                False,
                hl.struct(
                    is_de_novo=True,
                    p_de_novo=hl.missing(hl.tfloat64),
                    confidence=hl.missing(hl.tstr),
                    fail_reason={"min_de_novo_p", "min_proband_gq"},
                ),
            ),
            # 10. Autosomal locus with multi-allelic
            (
                hl.locus("chr1", 40000, reference_genome="GRCh38"),
                hl.literal(["C", "G", "A"]),
                hl.struct(GT=hl.call(0, 1), AD=[5, 30, 5], DP=40, GQ=99, PL=[99, 0, 1]),
                hl.struct(
                    GT=hl.call(0, 0), AD=[20, 0, 5], DP=25, GQ=60, PL=[0, 60, 120]
                ),
                hl.struct(
                    GT=hl.call(0, 0), AD=[25, 0, 5], DP=30, GQ=80, PL=[0, 80, 150]
                ),
                hl.literal(True),
                hl.literal(1e-5),
                True,
                None,
            ),
        ],
    )
    def test_default_get_de_novo_expr(
        self,
        locus,
        alleles,
        proband_expr,
        father_expr,
        mother_expr,
        is_xx_expr,
        freq_prior_expr,
        expected_exception,
        expected_result,
    ):
        """Test different scenarios of `default_get_de_novo_expr`."""
        if expected_exception:
            with pytest.raises(
                hl.utils.HailUserError,
                match="Must split multiallelic variants prior to running this function.",
            ):
                result_expr = default_get_de_novo_expr(
                    locus,
                    alleles,
                    proband_expr,
                    father_expr,
                    mother_expr,
                    is_xx_expr,
                    freq_prior_expr,
                )
                hl.eval(result_expr)
        else:
            result_expr = default_get_de_novo_expr(
                locus,
                alleles,
                proband_expr,
                father_expr,
                mother_expr,
                is_xx_expr,
                freq_prior_expr,
            )
            result = hl.eval(result_expr)
            expected_result = hl.eval(expected_result)

            assert result.is_de_novo == expected_result.is_de_novo
            assert (
                None if result.p_de_novo is None else round(result.p_de_novo, 3)
            ) == expected_result.p_de_novo
            assert result.confidence == expected_result.confidence
            assert result.fail_reason == expected_result.fail_reason

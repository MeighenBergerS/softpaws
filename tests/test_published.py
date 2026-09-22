"""Tests for the published effective-area loaders.

The values in ``REF`` were computed once from the pre-cleanup scripts'
private readers (examples 30, 31, 32 for KM3NeT and IceCube, 47 for the
angle-dependent table, 55 for P-ONE and TRIDENT, 81 for the TRIDENT 2025 map).
"""

import pathlib

import numpy as np
import pytest

from softpaws.data import published as pb

nan = np.nan

DATA_DIR = pathlib.Path(__file__).parents[1] / "src" / "softpaws" / "data" / "dataverse_files"

REF = {
    "published": {
        "query": [
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            7.9,
            8.5
        ],
        "arca230_trigger": [
            28668.843877541345,
            146494.35703576534,
            1415842.5778521623,
            6031109.826153472,
            18625498.5685665,
            47549232.162624404,
            nan
        ],
        "arca230_quoted": [
            55465.34480165745,
            285725.37290708127,
            2783687.171196877,
            12002177.483288998,
            36134073.008638546,
            86261427.23172519,
            nan
        ],
        "arca21": [
            nan,
            nan,
            nan,
            2964.9774914629656,
            335536.3720828121,
            2069618.7563499003,
            4458276.142229443
        ],
        "arca21_table": {
            "n": 54,
            "idx": [
                0,
                5,
                20,
                -1
            ],
            "log10_e": [
                5.650000007631338,
                6.150000140008875,
                7.650000007631338,
                10.949999981417943
            ],
            "aeff": [
                45.96467,
                8827.83,
                1338994.0,
                35343950.0
            ]
        }
    },
    "pone_allsky": {
        "n": 47,
        "log10_e": [
            3.038576779026217,
            3.893632958801498,
            6.989138576779025
        ],
        "aeff": [
            3786.8211326920236,
            67263.08070258188,
            10521954.844836766
        ]
    },
    "pone_120_150": {
        "n": 54,
        "log10_e": [
            2.993978680495534,
            3.7531835205992508,
            6.962921348314606
        ],
        "aeff": [
            3554.449547553965,
            41513.81603836808,
            289003.5793183315
        ]
    },
    "trident_up": {
        "n": 82,
        "log10_e": [
            3.0232482812160875,
            3.497830062709591,
            6.968915655931191
        ],
        "aeff": [
            35071.99317270653,
            204385.20356337345,
            3346924.0873478693
        ]
    },
    "trident_2025": {
        "shape": [
            12,
            12
        ],
        "cos": [
            0.917,
            0.75,
            0.583,
            0.417,
            0.25,
            0.083,
            -0.083,
            -0.25,
            -0.417,
            -0.583,
            -0.75,
            -0.917
        ],
        "log10_a_m2_row3": [
            0.031,
            0.423,
            0.877,
            1.425,
            1.706,
            2.16,
            2.286,
            2.458,
            2.536,
            2.802,
            2.959,
            3.053
        ]
    },
    "angle_dependent": [
        {
            "hi": 0.08,
            "lo": -0.25,
            "n": 55,
            "log10_e": [
                3.039644152815609,
                7.9701132756755175
            ],
            "aeff": [
                3013.393442940918,
                52784708.2741391
            ]
        },
        {
            "hi": -0.25,
            "lo": -0.5,
            "n": 51,
            "log10_e": [
                3.043759578594066,
                7.9430549489379505
            ],
            "aeff": [
                3504.0680734151297,
                1382005.436204364
            ]
        },
        {
            "hi": -0.5,
            "lo": -0.75,
            "n": 57,
            "log10_e": [
                3.0332297493956255,
                7.96830612777412
            ],
            "aeff": [
                3232.002560833973,
                15474.310486068638
            ]
        },
        {
            "hi": -0.75,
            "lo": -1.0,
            "n": 62,
            "log10_e": [
                3.0084789674501855,
                7.970723181055303
            ],
            "aeff": [
                2892.3602810362618,
                67.34726491513612
            ]
        }
    ],
    "icecube_dr2": {
        "log10_e": [
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0
        ],
        "aeff": [
            2770.8335402868174,
            94590.3696777405,
            822463.4724004329,
            2809662.8374168244,
            5936223.078126195,
            10856222.47745327
        ],
        "livetime_s": 428834817.50805515
    }
}


def close(actual, expected, rtol=1e-10):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                               rtol=rtol, atol=0.0)


def test_interpolate_aeff():
    table_e = np.array([4.0, 5.0, 6.0])
    table_a = np.array([1.0e4, 1.0e5, 1.0e6])
    out = pb.interpolate_aeff(np.array([3.5, 4.5, 6.0, 6.5]), table_e, table_a)
    assert np.isnan(out[0]) and np.isnan(out[3])
    close(out[1:3], [10.0**4.5, 1.0e6])


def test_arca230_trigger_level():
    log10_e, aeff = pb.arca230_trigger_level_aeff()
    assert np.all(np.diff(log10_e) >= 0.0) and np.all(aeff > 0.0)
    q = np.array(REF["published"]["query"])
    out = pb.interpolate_aeff(q, log10_e, aeff)
    expected = np.array(REF["published"]["arca230_trigger"])
    assert np.array_equal(np.isnan(out), np.isnan(expected))
    close(out[~np.isnan(out)], expected[~np.isnan(expected)])


def test_arca230_quoted_fit():
    q = np.array(REF["published"]["query"])
    out = pb.arca230_quoted_fit(q)
    expected = np.array(REF["published"]["arca230_quoted"])
    assert np.array_equal(np.isnan(out), np.isnan(expected))
    close(out[~np.isnan(out)], expected[~np.isnan(expected)])
    # a flat factor ~1.95 above the digitized trigger curve
    ratio = out / pb.interpolate_aeff(q, *pb.arca230_trigger_level_aeff())
    ratio = ratio[np.isfinite(ratio)]
    assert 1.8 < ratio.min() and ratio.max() < 2.1


def test_arca21_bright_track():
    log10_e, aeff = pb.arca21_bright_track_aeff()
    t = REF["published"]["arca21_table"]
    assert log10_e.size == t["n"] and np.all(aeff > 0.0)
    close(log10_e[t["idx"]], t["log10_e"])
    close(aeff[t["idx"]], t["aeff"])
    q = np.array(REF["published"]["query"])
    out = pb.interpolate_aeff(q, log10_e, aeff)
    expected = np.array(REF["published"]["arca21"])
    assert np.array_equal(np.isnan(out), np.isnan(expected))
    close(out[~np.isnan(out)], expected[~np.isnan(expected)])


def test_arca230_angle_dependent():
    bands = pb.arca230_angle_dependent_aeff()
    assert len(bands) == len(REF["angle_dependent"]) == 4
    for (hi, lo, log10_e, aeff), r in zip(bands, REF["angle_dependent"], strict=True):
        assert (hi, lo) == (r["hi"], r["lo"])
        assert log10_e.size == r["n"] and np.all(np.diff(log10_e) >= 0.0)
        close(log10_e[[0, -1]], r["log10_e"])
        close(aeff[[0, -1]], r["aeff"])
    assert bands[0][0] > bands[-1][0]


@pytest.mark.parametrize(
    "loader, key",
    [(pb.pone_allsky_aeff, "pone_allsky"), (lambda: pb.pone_band_aeff(120, 150), "pone_120_150"),
     (lambda: pb.trident_band_aeff(-1.0, -0.2), "trident_up")],
)
def test_pone_and_trident_bands(loader, key):
    log10_e, aeff = loader()
    r = REF[key]
    assert log10_e.size == r["n"] and np.all(np.diff(log10_e) >= 0.0) and np.all(aeff > 0.0)
    close(log10_e[[0, 10, -1]], r["log10_e"])
    close(aeff[[0, 10, -1]], r["aeff"])


def test_every_band_file_loads():
    for lo, hi in pb.PONE_ZENITH_BANDS_DEG:
        log10_e, aeff = pb.pone_band_aeff(lo, hi)
        assert log10_e.size > 10 and np.all(aeff > 0.0)
    for lo, hi in pb.TRIDENT_COS_BANDS:
        log10_e, aeff = pb.trident_band_aeff(lo, hi)
        assert log10_e.size > 10 and np.all(aeff > 0.0)
    with pytest.raises(ValueError):
        pb.trident_band_aeff(0.0, 0.5)


def test_trident_2025_map():
    cos_theta, log10_e, log10_aeff = pb.trident_2025_map()
    r = REF["trident_2025"]
    assert list(log10_aeff.shape) == r["shape"] == [12, 12]
    close(cos_theta, r["cos"])
    close(log10_e, 3.0 + 0.25 * (np.arange(12) + 0.5))
    close(log10_aeff[3] - 4.0, r["log10_a_m2_row3"])


@pytest.mark.skipif(not (DATA_DIR / "irfs").exists(), reason="the DR2 release is not on disk")
def test_icecube_dr2():
    r = REF["icecube_dr2"]
    grid = np.linspace(3.0, 8.0, 26)
    aeff, livetime_s = pb.icecube_dr2_aeff(DATA_DIR, grid)
    assert aeff.shape == grid.shape
    close(aeff[::5], r["aeff"])
    assert livetime_s == pytest.approx(r["livetime_s"], rel=1e-12)
    down, _ = pb.icecube_dr2_aeff(DATA_DIR, grid, "downgoing")
    assert down[10] < aeff[10]

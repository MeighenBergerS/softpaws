"""Tests for the effective-area engine, pinned against the pre-cleanup scripts.

Every value in ``REF`` was computed once from the ``pre-cleanup`` versions of
examples 30, 31 and 32 (``git show pre-cleanup:examples/...``), loaded by path
with their module-level grids shrunk to the ones given alongside each block.
The library runs the same code path, so the tolerance is ``rtol = 1e-10``.
"""

import numpy as np
import pytest

from softpaws.detectors import ARCA230, ICECUBE
from softpaws.response import effective_area as ea

inf = np.inf
nan = np.nan

REF = {
    "fit_reach_law": {
        "log10_e": [
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5
        ],
        "required": [
            0.55,
            nan,
            0.6,
            0.64,
            0.7,
            0.73
        ],
        "radius_km": 0.564,
        "result": [
            0.03251339769924339,
            214490.4410457425
        ]
    },
    "truncated_range_km": {
        "energy_mu": [
            3000.0,
            100000.0,
            10000000.0,
            1000000000.0
        ],
        "column": [
            1.0,
            3.18,
            30.0,
            inf
        ],
        "threshold": 1000.0,
        "running": [
            [
                0.9985957322101977,
                2.8087234575604567,
                3.6241453774275945,
                3.6241453774275962
            ],
            [
                0.9999999858156394,
                3.1796082908646524,
                11.684378598015886,
                11.684900342467552
            ],
            [
                0.9999999999999998,
                3.1799999945071393,
                20.931483041867324,
                21.14075784217623
            ],
            [
                1.0,
                3.179999999999975,
                27.095696843566273,
                29.422876096134427
            ]
        ],
        "frozen": [
            [
                0.998219608724054,
                2.771597344402677,
                3.516287379387316,
                3.5162873793873155
            ],
            [
                0.9999999133461179,
                3.178913305816757,
                10.801030554896352,
                10.801274995966974
            ],
            [
                0.9999999999999778,
                3.1799998934984774,
                18.545882305735162,
                18.609330613746792
            ],
            [
                1.0,
                3.179999999978001,
                23.022058645528546,
                23.42566508688659
            ]
        ]
    },
    "length_table": {
        "table_log10_e": [
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "threshold": 1000.0,
        "ell_last": 99.28046636510442,
        "n_ell": 101,
        "cumulative_last": [
            0.0,
            0.0,
            6.226118212198309,
            11.28883573658563,
            15.88431737025012,
            19.85226072417228,
            22.991976339639685,
            25.066051460529007,
            25.944092635161528
        ],
        "cumulative_mid": [
            0.0,
            0.0,
            6.226118212198309,
            11.28883573658563,
            15.88431737025011,
            19.85226072396943,
            22.99197630200008,
            25.06605090655485,
            25.944091370298082
        ],
        "lookup": [
            [
                0.47674772771365764,
                1.4869457737538958,
                2.970613333436998
            ],
            [
                0.9999666254664955,
                3.179315701478284,
                11.288835599559157
            ],
            [
                0.9999881641995144,
                3.179988082088299,
                19.83806348607097
            ],
            [
                0.9999960057568341,
                3.1799960056321788,
                24.733741989130863
            ]
        ]
    },
    "length_table_small": {
        "table_log10_e": [
            3.0,
            5.0,
            7.0
        ],
        "threshold": 1000.0,
        "n_ell": 11,
        "ell": [
            0.0,
            6.052085403056341,
            12.104170806112682,
            18.156256209169022,
            24.208341612225365,
            30.260427015281707,
            36.312512418338045,
            42.36459782139439,
            48.41668322445073,
            54.46876862750707,
            60.52085403056341
        ],
        "cumulative": [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                5.958062677864998,
                10.030580840899471,
                11.221507198091619,
                11.272414745949822,
                11.272702179258555,
                11.272702484413717,
                11.272702484495369,
                11.272702484495369,
                11.272702484495369,
                11.272702484495369
            ],
            [
                0.0,
                6.051936707682577,
                12.044342394436681,
                16.98859637751156,
                19.38662597409649,
                19.829013372721963,
                19.85150772583867,
                19.85183133019682,
                19.851832838924167,
                19.851832841529358,
                19.85183284153139
            ]
        ],
        "energy_mu": [
            3000.0,
            100000.0,
            10000000.0,
            1000000000.0
        ],
        "column": [
            1.0,
            3.18,
            30.0
        ],
        "lookup": [
            [
                0.2348544469585434,
                0.7468371413281681,
                2.689219953280229
            ],
            [
                0.9844644087236671,
                3.1305968197412617,
                11.272689810728751
            ],
            [
                0.9999754307211711,
                3.1799218696933242,
                19.809977020431877
            ],
            [
                0.9999754307211711,
                3.1799218696933242,
                19.809977020431877
            ]
        ]
    },
    "ic_upgoing_columns_full": {
        "columns": [
            11341738.562668629,
            1143355949.6517127,
            2622472213.2399797,
            4172672272.325895,
            7026591562.142147,
            9913822230.603045
        ],
        "weights": [
            0.9999619230641713,
            0.9632341289777294,
            0.8601240001761764,
            0.6977374840888118,
            0.48726562528085715,
            0.24321332236333779
        ],
        "cos": [
            0.008726535498373935,
            0.26866338189733796,
            0.5100849971533499,
            0.7163535463005786,
            0.8732538064157839,
            0.9699728242713747
        ]
    },
    "ic_upgoing_columns": {
        "n_dec": 8,
        "columns": [
            11341738.562668629,
            962793610.5110613,
            2064232373.8499236,
            3454547502.6101604,
            4720971457.313929,
            7846462320.808863,
            10009905528.824871,
            10954255642.213196
        ],
        "weights": [
            0.9999619230641713,
            0.9735219372611233,
            0.8993398491364701,
            0.7810535929843699,
            0.6244639980705103,
            0.4372503124700897,
            0.2285936081156897,
            0.008726535498373897
        ],
        "cos": [
            0.008726535498373935,
            0.22859360811568963,
            0.4372503124700897,
            0.6244639980705101,
            0.7810535929843698,
            0.8993398491364701,
            0.9735219372611233,
            0.9999619230641713
        ]
    },
    "ic_target_volume_cm3": {
        "lengths": [
            0.0,
            2.0,
            8.0
        ],
        "scalar": [
            1539380400258998.2,
            5267790582561498.0,
            1.6453021129468996e+16
        ],
        "array": [
            [
                785398163397448.2,
                785398163397448.2,
                785398163397448.2,
                785398163397448.2,
                785398163397448.2,
                785398163397448.2,
                785398163397448.2,
                785398163397448.2
            ],
            [
                4506345741738008.0,
                5105524745524681.0,
                5529817882122497.0,
                5758417560488571.0,
                5780113114671249.0,
                5593840581622480.0,
                5208734878586696.0,
                4643681821255623.0
            ],
            [
                1.7842846864973332e+16,
                2.1918994087792884e+16,
                2.504501376812197e+16,
                2.7067604017502372e+16,
                2.788757579155338e+16,
                2.7464717180466656e+16,
                2.5819765425289772e+16,
                2.3033389951131616e+16
            ]
        ],
        "mean": [
            1539380400258998.5,
            5296666270361114.0,
            1.6568523880667466e+16
        ],
        "mean_default": [
            1000000000000000.0,
            3812090794465282.5,
            1.2248363177861132e+16
        ]
    },
    "ic_required_radius_km": {
        "ratio": [
            None,
            0.6,
            1.3
        ],
        "lengths": [
            0.0,
            2.0,
            8.0
        ],
        "result": [
            nan,
            0.3970797811355878,
            0.6807220504657544
        ]
    },
    "ic_aeff": {
        "log10_e": [
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0
        ],
        "n_dec": 8,
        "threshold": 1000.0,
        "length": [
            0.0,
            5.115620297135499,
            10.27151649907261,
            15.156919001653774,
            19.77939098029872,
            24.11754347234484
        ],
        "regenerated": [
            0.0,
            160535.38991933572,
            1242930.9111574506,
            3896941.819059524,
            8100463.545914803,
            16689452.771451745
        ],
        "regenerated_reach": [
            0.0,
            119173.16565359583,
            1060453.9401080955,
            3733892.9326609243,
            8524870.43263102,
            18985081.02941055
        ],
        "tau": [
            0.0,
            15077.741143331632,
            177393.05053254083,
            766284.5543094596,
            2101934.491865691,
            4634259.306567323
        ]
    },
    "ic_aeff_31": {
        "regenerated": [
            2426.0540256947443,
            161899.27714187227,
            1224264.5714690285,
            3556588.2465704484,
            6091289.074086119,
            8586024.233033804
        ],
        "regenerated_reach": [
            1266.8322453712933,
            119548.89205151254,
            1040649.0472787957,
            3401652.2893898827,
            6415229.74989586,
            9793775.480904583
        ],
        "tau": [
            422.8831119125318,
            15255.014204461715,
            176377.87807688836,
            738079.3307967404,
            1902112.5509755262,
            3748281.0686707236
        ]
    },
    "projected_area": {
        "theta": [
            0.0,
            30.0,
            90.0,
            150.0,
            180.0
        ],
        "scalar": [
            1.6794263175707245,
            2.1079138548003997,
            1.3069760000000001,
            2.1079138548003997,
            1.6794263175707247
        ],
        "column": [
            [
                0.2827433388230814,
                0.4344629141716194,
                0.3792,
                0.4344629141716194,
                0.28274333882308145
            ],
            [
                1.1309733552923256,
                1.3586516566864777,
                0.7584000000000001,
                1.3586516566864777,
                1.1309733552923256
            ]
        ],
        "height": [
            1.6794263175707245,
            2.2816258548003994,
            1.6544,
            2.2816258548003994,
            1.6794263175707247
        ]
    },
    "arca_aeff": {
        "log10_e": [
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0
        ],
        "n_zenith": 12,
        "threshold_gev": 1000.0,
        "depth_km": 3.184,
        "mu": [
            225036.15469860306,
            1568650.647789866,
            5140939.07069159,
            12284837.30306109,
            27119988.412963018,
            57928895.18447246,
            116600622.29352862
        ],
        "tau": [
            23543.335843710727,
            238524.22719448264,
            966286.0878246971,
            2677787.0954962973,
            6158474.08629552,
            12928941.300135158,
            25111868.401209507
        ],
        "mu_reach": [
            6445.299653050164,
            138020.74158352742,
            831570.2314715607,
            2993246.483472356,
            8945279.734650372,
            24470898.085726727,
            61054036.58396702
        ],
        "mu_frozen": [
            216650.92734519704,
            1487860.733185716,
            4851500.536936233,
            11550651.249244343,
            25295941.432109132,
            53363954.74553931,
            105551066.95883864
        ]
    },
    "arca_aeff_30": {
        "height_km": 0.7,
        "mu": [
            239994.42187341573,
            1677469.40528311,
            5525952.036918058,
            13254440.879355364,
            29281582.281074468,
            62505332.638072364,
            125750773.0942566
        ],
        "tau_reach": [
            119.247356446072,
            14139.437584842146,
            121158.94619438774,
            533993.2412404019,
            1734682.0238827222,
            4826707.580571111,
            11907677.209211558
        ],
        "band": [
            244355.42406424103,
            2024864.5981322094,
            7412456.123842579,
            15132780.31703001,
            21564594.531909782,
            29484939.376717586,
            42157686.664609045
        ],
        "untruncated": [
            263607.7223519179,
            2159497.027378695,
            8627730.631474618,
            25998537.349178553,
            70368646.69517426,
            176631626.2970829,
            401070256.2734603
        ],
        "required": [
            nan,
            0.44101809337912157,
            0.6518662768338076
        ],
        "volume": [
            47.77548067550607,
            9.25967861996183,
            2.6864330377027645,
            0.9598617559017879,
            0.3881023144858725,
            0.16981252997329133,
            0.08161606430696126
        ]
    },
    "arca_aeff_31": {
        "mu": [
            211631.8617210904,
            1531272.3348126914,
            5024525.632150112,
            11921524.093777265,
            26010067.703297712,
            54696759.938741654,
            107970390.80251962
        ],
        "tau_reach": [
            78.10431429537871,
            12340.821564367592,
            107983.21346988124,
            476065.8232166802,
            1534593.55192124,
            4220311.566682123,
            10259757.683989873
        ]
    },
    "required_footprint_radius_km": {
        "ratio": [
            None,
            0.8,
            1.4
        ],
        "n_zenith": 12,
        "result": [
            nan,
            0.4422537941359432,
            0.6493234736401168
        ]
    }
}

THRESHOLD_GEV = 1000.0
DEPTH_KM = ARCA230.depth_km


#: The deterministic range is integrated on a fixed lattice rather than on a
#: grid refined to each descent, which converged it and moved these pre-cleanup
#: pins by up to 2e-5. Anything the range does not reach still holds at
#: ``rtol = 1e-10``.
QUADRATURE_RTOL = 5.0e-5


def close(actual, expected, rtol=1e-10):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0)


def test_grids_and_defaults():
    assert ea.IC_LOG10_E.size == 26 and ea.IC_LOG10_E[0] == 3.0 and ea.IC_LOG10_E[-1] == 8.0
    assert ea.ARCA_LOG10_E.size == 31 and ea.ARCA_LOG10_E[0] == 4.0
    assert ea.IC_FIT_BAND == (5.0, 7.8) and ea.ARCA_FIT_BAND == (4.0, 7.5)
    assert ea.N_DEC == 60 and ea.N_ZENITH == 90
    assert ea.default_cross_section() is ea.default_cross_section()


def test_fit_reach_law():
    r = REF["fit_reach_law"]
    reach, pivot = ea.fit_reach_law(np.array(r["log10_e"]), np.array(r["required"]), r["radius_km"])
    close([reach, pivot], r["result"])


def test_truncated_range_km():
    r = REF["truncated_range_km"]
    energy, column = np.array(r["energy_mu"]), np.array(r["column"])
    close(ea.truncated_range_km(energy, column, r["threshold"]), r["running"])
    close(ea.truncated_range_km(energy, column, r["threshold"], "frozen"), r["frozen"])
    # inf column returns the untruncated length, and the table is (n, m)
    assert ea.truncated_range_km(energy, column, r["threshold"]).shape == (4, 4)


def test_first_passage_length_table_small():
    r = REF["length_table_small"]
    ell, cumulative = ea.first_passage_length_table(
        10.0 ** np.array(r["table_log10_e"]), r["threshold"], n_ell=r["n_ell"]
    )
    close(ell, r["ell"], rtol=QUADRATURE_RTOL)
    close(cumulative, r["cumulative"], rtol=QUADRATURE_RTOL)
    lookup = ea.truncated_range_from_table_km(
        np.array(r["energy_mu"]), np.array(r["column"]), np.array(r["table_log10_e"]),
        ell, cumulative,
    )
    close(lookup, r["lookup"], rtol=QUADRATURE_RTOL)


@pytest.mark.slow
def test_first_passage_length_table():
    r = REF["length_table"]
    ell, cumulative = ea.first_passage_length_table(
        10.0 ** np.array(r["table_log10_e"]), r["threshold"], n_ell=r["n_ell"]
    )
    assert ell.size == r["n_ell"]
    close(ell[-1], r["ell_last"], rtol=QUADRATURE_RTOL)
    close(cumulative[:, -1], r["cumulative_last"], rtol=QUADRATURE_RTOL)
    close(cumulative[:, 50], r["cumulative_mid"], rtol=QUADRATURE_RTOL)
    energy = np.array(REF["truncated_range_km"]["energy_mu"])
    column = np.array(REF["truncated_range_km"]["column"][:3])
    grid = np.array(r["table_log10_e"])
    close(ea.truncated_range_from_table_km(energy, column, grid, ell, cumulative),
          r["lookup"], rtol=QUADRATURE_RTOL)


def test_ic_upgoing_columns():
    r = REF["ic_upgoing_columns_full"]
    columns, weights, cos_theta = ea.ic_upgoing_columns()
    assert columns.shape == (60,)
    close(columns[::10], r["columns"])
    close(weights[::10], r["weights"])
    close(cos_theta[::10], r["cos"])
    r = REF["ic_upgoing_columns"]
    columns, weights, cos_theta = ea.ic_upgoing_columns(r["n_dec"])
    close(columns, r["columns"])
    close(weights, r["weights"])
    close(cos_theta, r["cos"])


def test_ic_target_volume():
    r = REF["ic_target_volume_cm3"]
    lengths = np.array(r["lengths"])
    n_dec = REF["ic_upgoing_columns"]["n_dec"]
    _, _, cos_theta = ea.ic_upgoing_columns(n_dec)
    close(ea.ic_target_volume_cm3(lengths, 0.7, 0.3), r["scalar"])
    close(ea.ic_target_volume_cm3(lengths, np.array([0.5, 0.7, 0.9]), cos_theta), r["array"])
    close(ea.ic_mean_target_volume_cm3(lengths, 0.7, n_dec), r["mean"])
    close(ea.ic_mean_target_volume_cm3(lengths, n_dec=n_dec), r["mean_default"])
    # the body alone is the instrumented volume
    body = ea.ic_target_volume_cm3(np.array([0.0]))[0] / 1.0e15
    assert body == pytest.approx(ICECUBE.detector_volume_km3())


def test_ic_required_radius_km():
    r = REF["ic_required_radius_km"]
    ratio = np.array([np.nan if x is None else x for x in r["ratio"]])
    n_dec = REF["ic_upgoing_columns"]["n_dec"]
    out = ea.ic_required_radius_km(ratio, np.array(r["lengths"]), n_dec)
    assert np.isnan(out[0])
    close(out[1:], r["result"][1:])


def test_ic_effective_area():
    r = REF["ic_aeff"]
    grid, length = np.array(r["log10_e"]), np.array(r["length"])
    kw = dict(log10_e=grid, n_dec=r["n_dec"])
    close(ea.ic_effective_area_regenerated(length, r["threshold"], **kw), r["regenerated"])
    close(
        ea.ic_effective_area_regenerated(length, r["threshold"], 0.02, 2.0e6, **kw),
        r["regenerated_reach"],
    )
    close(ea.ic_effective_area_tau_channel(length, r["threshold"], **kw), r["tau"])


def test_ic_effective_area_without_subthreshold_mask():
    """Example 31's construction keeps the body of a sub-threshold rung."""
    r = REF["ic_aeff"]
    grid, length = np.array(r["log10_e"]), np.array(r["length"])
    kw = dict(log10_e=grid, n_dec=60, mask_subthreshold=False)
    r31 = REF["ic_aeff_31"]
    close(ea.ic_effective_area_regenerated(length, r["threshold"], **kw), r31["regenerated"])
    close(
        ea.ic_effective_area_regenerated(length, r["threshold"], 0.02, 2.0e6, **kw),
        r31["regenerated_reach"],
    )
    close(ea.ic_effective_area_tau_channel(length, r["threshold"], **kw), r31["tau"])
    # the mask can only lower the area
    masked = ea.ic_effective_area_regenerated(length, r["threshold"], log10_e=grid, n_dec=60)
    assert np.all(masked <= np.array(r31["regenerated"]) * (1.0 + 1e-12))


def test_cylinder_projected_area_km2():
    r = REF["projected_area"]
    theta = np.array(r["theta"])
    close(ea.cylinder_projected_area_km2(theta, 0.517, 2), r["scalar"])
    close(ea.cylinder_projected_area_km2(theta[None, :], np.array([[0.3], [0.6]]), 1), r["column"])
    close(ea.cylinder_projected_area_km2(theta, 0.517, 2, 0.8), r["height"])
    # overhead is the footprint, the horizon the side
    assert ea.cylinder_projected_area_km2(0.0, 0.5, 1, 0.6) == pytest.approx(np.pi * 0.25)
    assert ea.cylinder_projected_area_km2(90.0, 0.5, 1, 0.6) == pytest.approx(0.6)


def test_arca_effective_area():
    r = REF["arca_aeff"]
    kw = dict(log10_e=np.array(r["log10_e"]), n_zenith=r["n_zenith"])
    args = (r["threshold_gev"], r["depth_km"])
    close(ea.arca_effective_area(0.517, 2, *args, "mu", **kw), r["mu"], rtol=QUADRATURE_RTOL)
    close(ea.arca_effective_area(0.517, 2, *args, "tau", **kw), r["tau"], rtol=QUADRATURE_RTOL)
    close(ea.arca_effective_area(0.221, 1, *args, "mu", reach_km=0.035, pivot_gev=1.0e6, **kw),
          r["mu_reach"], rtol=QUADRATURE_RTOL)
    close(ea.arca_effective_area(0.517, 2, *args, "mu", "frozen", **kw), r["mu_frozen"])


def test_arca_effective_area_example_30_options():
    """Example 30: taller block, no sub-threshold mask, zenith band, no truncation."""
    r = REF["arca_aeff"]
    r30 = REF["arca_aeff_30"]
    kw = dict(log10_e=np.array(r["log10_e"]), n_zenith=r["n_zenith"], height_km=r30["height_km"],
              mask_subthreshold=False)
    args = (r["threshold_gev"], r["depth_km"])
    close(ea.arca_effective_area(0.517, 2, *args, "mu", **kw), r30["mu"], rtol=QUADRATURE_RTOL)
    close(ea.arca_effective_area(0.221, 1, *args, "tau", reach_km=0.035, pivot_gev=1.0e6, **kw),
          r30["tau_reach"], rtol=QUADRATURE_RTOL)
    close(ea.arca_effective_area(0.517, 2, *args, "mu", cos_range=(-0.5, 0.0), **kw),
          r30["band"], rtol=QUADRATURE_RTOL)
    close(ea.arca_effective_area(0.517, 2, *args, "mu", truncate=False, **kw),
          r30["untruncated"], rtol=QUADRATURE_RTOL)
    required = ea.required_footprint_radius_km(
        np.array([np.nan, 0.8, 1.4]), 0.517, 2, r30["height_km"], r["n_zenith"]
    )
    assert np.isnan(required[0])
    close(required[1:], r30["required"][1:])
    close(ea.effective_volume_km3(np.full(len(r["log10_e"]), 1.0e6), np.array(r["log10_e"])),
          r30["volume"])


def test_arca_effective_area_example_31_table_range():
    """Example 31 drives the same engine with the frozen-kernel table range."""
    r = REF["arca_aeff"]
    t = REF["length_table"]
    r31 = REF["arca_aeff_31"]
    grid = np.array(t["table_log10_e"])
    ell, cumulative = ea.first_passage_length_table(10.0**grid, t["threshold"], n_ell=t["n_ell"])
    kw = dict(
        log10_e=np.array(r["log10_e"]), n_zenith=r["n_zenith"], mask_subthreshold=False,
        truncated_range=lambda e, x: ea.truncated_range_from_table_km(e, x, grid, ell, cumulative),
    )
    args = (r["threshold_gev"], r["depth_km"])
    close(ea.arca_effective_area(0.517, 2, *args, "mu", **kw), r31["mu"],
          rtol=QUADRATURE_RTOL)
    close(ea.arca_effective_area(0.221, 1, *args, "tau", reach_km=0.035, pivot_gev=1.0e6, **kw),
          r31["tau_reach"], rtol=QUADRATURE_RTOL)


test_arca_effective_area_example_31_table_range = pytest.mark.slow(
    test_arca_effective_area_example_31_table_range
)


def test_required_footprint_radius_km():
    r = REF["required_footprint_radius_km"]
    out = ea.required_footprint_radius_km(
        np.array([np.nan, 0.8, 1.4]), 0.517, 2, n_zenith=r["n_zenith"]
    )
    assert np.isnan(out[0])
    close(out[1:], r["result"][1:])
    # ratio one returns the instrumented radius
    assert ea.required_footprint_radius_km(np.array([1.0]), 0.517, 2)[0] == pytest.approx(0.517)

"""Tests for the staged hydraulic-depth and bathymetry-burn workflow."""

from __future__ import annotations

import numpy as np

import arc.cross_section as cross_section_module
import arc.Automated_Rating_Curve_Generator as generator
from arc.cross_section import CrossSection


def test_bathymetry_nan_fill_averages_non_nan_neighbors() -> None:
    """An eligible NaN should receive the mean of its non-NaN neighbors."""
    bathymetry = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )

    returned = generator._fill_bathymetry_nan_cells(bathymetry)

    assert returned is bathymetry
    assert bathymetry[1, 1] == 4.5


def test_bathymetry_nan_fill_accepts_four_neighbors_without_propagating() -> None:
    """Four original neighbors fill a cell without enabling another fill."""
    bathymetry = np.asarray(
        [
            [4.0, 1.0, 2.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, 3.0, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    generator._fill_bathymetry_nan_cells(bathymetry)

    assert bathymetry[1, 1] == 2.5
    # Cell [1, 2] originally has three valid neighbors. The newly filled
    # [1, 1] would be its fourth, but a synchronous one-pass fill cannot use it.
    assert np.isnan(bathymetry[1, 2])


def test_bathymetry_nan_fill_rejects_three_neighbors() -> None:
    """A NaN supported by fewer than four values must remain NaN."""
    bathymetry = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [3.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    generator._fill_bathymetry_nan_cells(bathymetry)

    assert np.isnan(bathymetry[1, 1])


def _build_cross_section(*, use_bank_elevations: bool) -> CrossSection:
    """Create a small cross section with explicit profiles and raster indices."""
    params = {
        "d_x_section_distance": 10.0,
        "b_FindBanksBasedOnLandCover": False,
        "i_lc_water_value": 80,
        "d_bathymetry_trapzoid_height": 0.1,
        "b_bathy_use_banks": use_bank_elevations,
        "d_degree_manipulation": 0.0,
        "d_degree_interval": 0.0,
        "i_boundary_number": 0,
        "nrows": 7,
        "ncols": 7,
    }
    x_section = CrossSection(
        1.0,
        1.0,
        np.full((7, 7), 10.0, dtype=np.float64),
        np.zeros((7, 7), dtype=np.uint8),
        np.zeros((7, 7), dtype=np.int64),
        params,
    )
    x_section.xs1_n = 4
    x_section.xs2_n = 4
    x_section.d_ordinate_dist = 1.0
    x_section.da_xs_profile1[:4] = [10.0, 10.5, 11.0, 12.0]
    x_section.da_xs_profile2[:4] = [10.0, 10.4, 11.0, 12.0]
    x_section.ia_xc_row1_index_main = np.array([3, 3, 3, 3])
    x_section.ia_xc_column1_index_main = np.array([3, 2, 1, 0])
    x_section.ia_xc_row2_index_main = np.array([3, 3, 3, 3])
    x_section.ia_xc_column2_index_main = np.array([3, 4, 5, 6])
    return x_section


def _bank_result(**updates) -> dict:
    """Return a valid staged bank result for the test cross section."""
    result = {
        "function_used": "test",
        "i_bank_1_index": 2,
        "i_bank_2_index": 2,
        "bank_elev_1": 11.0,
        "bank_elev_2": 11.0,
        "smoothed_bank_elevation": 11.0,
        "is_valid": True,
    }
    result.update(updates)
    return result


def test_hydraulic_depth_is_solved_without_mutating_cross_section() -> None:
    """The separate hydraulic pass should only return a depth."""
    x_section = _build_cross_section(use_bank_elevations=False)
    profile1_before = x_section.da_xs_profile1.copy()
    profile2_before = x_section.da_xs_profile2.copy()

    depth = x_section.calculate_hydraulic_bathymetry_depth(
        5.0,
        0.002,
        _bank_result(),
    )

    assert 0.0 < depth < 25.0
    np.testing.assert_array_equal(x_section.da_xs_profile1, profile1_before)
    np.testing.assert_array_equal(x_section.da_xs_profile2, profile2_before)


def test_bathymetry_burn_consumes_staged_depth_without_solving(monkeypatch) -> None:
    """The final burn must not call either hydraulic-depth solver."""
    x_section = _build_cross_section(use_bank_elevations=True)
    output_bathymetry = np.full((7, 7), 10.0, dtype=np.float64)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("hydraulic solver was called during bathymetry burn")

    monkeypatch.setattr(
        cross_section_module,
        "find_depth_of_bathymetry",
        fail_if_called,
    )
    monkeypatch.setattr(
        cross_section_module,
        "find_depth_of_bathymetry_triangle",
        fail_if_called,
    )

    result = _bank_result(bathymetry_depth=1.5)
    _, _, _, applied_depth, bottom_elevation = (
        x_section.Calculate_Bathymetry_Based_on_RiverBank_Elevations(
            output_bathymetry,
            bank_search_result=result,
        )
    )

    assert applied_depth == 1.5
    assert bottom_elevation == 9.5
    assert np.nanmin(output_bathymetry) == 9.5


def test_staged_target_preserves_previous_invalid_bank_fallback() -> None:
    """A reach-smoothed one-cell fallback must still accept a target depth."""
    x_section = _build_cross_section(use_bank_elevations=True)
    output_bathymetry = np.full((7, 7), 10.0, dtype=np.float64)
    result = _bank_result(
        is_valid=False,
        bathymetry_depth=1.5,
        bathymetry_should_apply=True,
    )

    _, _, _, applied_depth, bottom_elevation = (
        x_section.Calculate_Bathymetry_Based_on_RiverBank_Elevations(
            output_bathymetry,
            bank_search_result=result,
        )
    )

    assert applied_depth == 1.5
    assert bottom_elevation == 9.5
    assert output_bathymetry[3, 3] == 9.5


def test_hydraulic_solver_preserves_previous_invalid_bank_fallback() -> None:
    """The separate solve retains the former one-cell fallback bank geometry."""
    x_section = _build_cross_section(use_bank_elevations=False)

    depth = x_section.calculate_hydraulic_bathymetry_depth(
        5.0,
        0.002,
        _bank_result(is_valid=False),
    )

    assert 0.0 < depth < 25.0


def test_bathymetry_burn_skips_missing_staged_depth() -> None:
    """A valid bank result alone is insufficient to trigger a bathymetry burn."""
    x_section = _build_cross_section(use_bank_elevations=False)
    output_bathymetry = np.full((7, 7), 10.0, dtype=np.float64)
    profile_before = x_section.da_xs_profile1.copy()

    _, _, _, applied_depth, _ = x_section.Calculate_Bathymetry_Based_on_WSE_or_LC(
        output_bathymetry,
        bank_search_result=_bank_result(),
    )

    assert applied_depth == 0.0
    np.testing.assert_array_equal(x_section.da_xs_profile1, profile_before)
    np.testing.assert_array_equal(output_bathymetry, 10.0)


def test_staging_pass_solves_only_depths_without_power_law_targets(
    monkeypatch,
) -> None:
    """Target depths are copied; only non-target cells use the hydraulic solve."""

    class FakeCrossSection:
        def __init__(self):
            self.hydraulic_calls = []

        @staticmethod
        def _is_valid_bathymetry_target(value):
            return value is not None and np.isfinite(value) and value > 0.0

        def calculate_hydraulic_bathymetry_depth(self, flow, slope, bank_result):
            self.hydraulic_calls.append((flow, slope, bank_result))
            return 1.25

    fake_cross_section = FakeCrossSection()
    sampled_records = [
        {
            "bank_search_result": {
                "is_valid": True,
                "network_reach_bank_elevation_grade": 0.004,
            }
        },
        {
            "bank_search_result": {
                "is_valid": True,
                "network_reach_bank_elevation_grade": 0.006,
            }
        },
    ]
    cell_inputs = [
        (3.0, 0.001, 2.5, 8.0),
        (4.0, 0.002, None, None),
    ]

    monkeypatch.setattr(generator, "_CELL_COMIDS", np.array([10, 20]))
    monkeypatch.setattr(generator, "_CELL_ROWS", np.array([1, 2]))
    monkeypatch.setattr(generator, "_CELL_COLS", np.array([3, 4]))
    monkeypatch.setattr(generator, "_CELL_REACH_INFLECT_BANK_INDEX", None)
    monkeypatch.setattr(generator, "get_cross_section", lambda *_args: fake_cross_section)
    monkeypatch.setattr(
        generator,
        "_replay_precomputed_cross_section",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generator,
        "_get_cell_bathymetry_inputs",
        lambda entry, *_args: cell_inputs[entry],
    )
    # This test isolates candidate-depth staging. Reach aggregation and network
    # monotonicity are covered independently below.
    monkeypatch.setattr(
        generator,
        "_smooth_reach_bathymetry_depths",
        lambda *_args, **_kwargs: None,
    )

    generator._stage_cross_section_bathymetry_depths(
        sampled_records,
        {"dx": 1.0, "dy": 1.0},
        quiet=True,
    )

    assert sampled_records[0]["bank_search_result"]["bathymetry_depth"] == 2.5
    assert (
        sampled_records[0]["bank_search_result"]["bathymetry_depth_source"]
        == "drainage_area_power_law"
    )
    assert sampled_records[1]["bank_search_result"]["bathymetry_depth"] == 1.25
    assert (
        sampled_records[1]["bank_search_result"]["bathymetry_depth_source"]
        == "baseflow_manning"
    )
    assert len(fake_cross_section.hydraulic_calls) == 1
    assert fake_cross_section.hydraulic_calls[0][1] == 0.006
    assert (
        sampled_records[1]["bank_search_result"][
            "bathymetry_depth_original_slope"
        ]
        == 0.002
    )
    assert (
        sampled_records[1]["bank_search_result"][
            "bathymetry_depth_smoothed_bank_slope"
        ]
        == 0.006
    )


def test_smoothed_bank_grade_replaces_existing_cell_slope() -> None:
    """The network bank-surface grade should be the authoritative slope."""
    replaced = generator._replace_slope_with_smoothed_bank_grade(
        0.02,
        {"network_reach_bank_elevation_grade": 0.003},
    )

    assert replaced == 0.003


def test_flat_smoothed_bank_grade_uses_only_numerical_slope_floor() -> None:
    """A flat reach should not reintroduce the former 0.001 minimum grade."""
    replaced = generator._replace_slope_with_smoothed_bank_grade(
        0.02,
        {"network_reach_bank_elevation_grade": 0.0},
    )

    assert replaced == generator.MIN_SLOPE
    assert replaced < 0.001


def test_missing_or_invalid_smoothed_grade_retains_existing_slope() -> None:
    """Slope replacement should wait until smoothing produced a valid grade."""
    assert generator._replace_slope_with_smoothed_bank_grade(0.02, None) == 0.02
    assert (
        generator._replace_slope_with_smoothed_bank_grade(
            0.02,
            {"network_reach_bank_elevation_grade": np.nan},
        )
        == 0.02
    )
    assert (
        generator._replace_slope_with_smoothed_bank_grade(
            0.02,
            {"network_reach_bank_elevation_grade": None},
        )
        == 0.02
    )


def test_filtered_reach_depth_uses_interquartile_median() -> None:
    """Reach candidates outside Q25-Q75 must not influence the median."""
    candidate_depths = [1.0, 2.0, 3.0, 4.0, 20.0, 0.0, 5.0, 9.0]
    reach_ids = np.array([100, 100, 100, 100, 100, 100, 200, 200])
    sampled_records = [
        {
            "bank_search_result": {
                "bathymetry_depth": depth,
                "bathymetry_should_apply": depth > 0.0,
            }
        }
        for depth in candidate_depths
    ]

    medians, statistics = generator._compute_filtered_reach_median_depths(
        sampled_records,
        reach_ids,
    )

    # Reach 100 has Q25=2 and Q75=4; [2, 3, 4] has a median of 3.
    assert medians[100] == 3.0
    assert statistics[100] == {
        "q25": 2.0,
        "q75": 4.0,
        "median": 3.0,
        "candidate_count": 5,
        "retained_count": 3,
    }
    # Two interpolated quartiles retain no observations, so both values are
    # used by the documented small-sample fallback.
    assert medians[200] == 7.0


def test_reach_depth_smoothing_assigns_network_depth_to_every_section(
    monkeypatch,
) -> None:
    """All sections in a reach should receive its constrained median depth."""
    sampled_records = [
        {
            "bank_search_result": {
                "bathymetry_depth": 2.0,
                "bathymetry_should_apply": True,
                "bathymetry_depth_source": "baseflow_manning",
            }
        },
        {
            "bank_search_result": {
                "bathymetry_depth": 4.0,
                "bathymetry_should_apply": True,
                "bathymetry_depth_source": "baseflow_manning",
            }
        },
        {
            "bank_search_result": {
                "bathymetry_depth": 1.0,
                "bathymetry_should_apply": True,
                "bathymetry_depth_source": "baseflow_manning",
            }
        },
    ]
    graph = generator.nx.DiGraph()
    graph.add_edge(10, 20)

    monkeypatch.setattr(generator, "_CELL_SOURCE_STREAM_IDS", np.array([10, 10, 20]))
    monkeypatch.setattr(
        generator,
        "_build_reach_network_graph",
        lambda *_args: (graph, {10: 20, 20: None}),
    )

    generator._smooth_reach_bathymetry_depths(sampled_records, {})

    # Reach 10's two-value fallback median is 3. Reach 20's local median is 1,
    # so downstream monotonicity raises it to 3.
    assert [
        record["bank_search_result"]["bathymetry_depth"]
        for record in sampled_records
    ] == [3.0, 3.0, 3.0]
    assert (
        sampled_records[2]["bank_search_result"][
            "bathymetry_depth_monotonic_adjustment"
        ]
        == 2.0
    )

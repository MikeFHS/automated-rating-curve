from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString

from arc.Automated_Rating_Curve_Generator import (
    MIN_SLOPE,
    _anchor_interpolated_bank_surface_to_cell_observations,
    _exclude_thalweg_equal_bank_elevations,
    _estimate_network_smoothed_reach_min_bank_elevations,
    _interpolate_reach_bank_elevation_surface,
    _measure_reach_geometry_length,
    _order_reach_stream_cells_from_network,
    _reconstruct_reach_bank_width_with_fallback,
)


def test_geographic_reach_length_is_measured_in_meters() -> None:
    """A geographic flowline must not pass its degree length to the graph."""
    geometry = LineString([(-122.8, 44.5), (-122.8, 44.55)])
    geographic_crs = CRS.from_epsg(4269)

    measured_length = _measure_reach_geometry_length(
        geometry,
        geographic_crs,
    )
    expected_length = abs(
        geographic_crs.get_geod().geometry_length(geometry)
    )

    assert geometry.length == pytest.approx(0.05)
    assert measured_length == pytest.approx(expected_length)
    assert measured_length > 5_000.0


def test_projected_reach_length_converts_native_units_to_meters() -> None:
    """Projected flowlines should honor their CRS linear-unit conversion."""
    geometry = LineString([(0.0, 0.0), (1000.0, 0.0)])
    feet_crs = CRS.from_epsg(2263)

    measured_length = _measure_reach_geometry_length(geometry, feet_crs)
    expected_length = (
        geometry.length * feet_crs.axis_info[0].unit_conversion_factor
    )

    assert measured_length == pytest.approx(expected_length)
    assert measured_length == pytest.approx(304.8006096)


def test_excludes_thalweg_equal_banks_before_reach_statistics() -> None:
    """Banks at or below the thalweg must not enter reach statistics."""
    filtered = _exclude_thalweg_equal_bank_elevations(
        np.asarray([100.0, 101.5, 99.00000001, 97.5, np.nan]),
        np.asarray([100.0, 100.0, 99.0, 98.0, 98.0]),
    )

    assert np.isnan(filtered[0])
    assert filtered[1] == pytest.approx(101.5)
    assert np.isnan(filtered[2])
    assert np.isnan(filtered[3])
    assert np.isnan(filtered[4])


class _WidthReconstructionCrossSection:
    """Minimal cross-section double for deterministic width-retry tests."""

    def __init__(self, successful_target: float | None) -> None:
        self.d_ordinate_dist = 5.0
        self.successful_target = successful_target
        self.attempted_targets: list[float] = []
        self.one_cell_fallback_calls = 0

    def build_bank_search_result_from_target_width(
        self,
        existing_result: dict,
        target_width: float,
        function_used: str,
    ) -> dict:
        self.attempted_targets.append(float(target_width))
        if self.successful_target is not None and np.isclose(
            target_width,
            self.successful_target,
        ):
            return {
                "is_valid": True,
                "i_bank_1_index": 2,
                "i_bank_2_index": 3,
                "resolved_width": float(target_width),
                "reach_top_width_filter_applied": True,
                "function_used": function_used,
            }
        return dict(existing_result)

    def get_top_width_from_bank_search_result(self, result: dict) -> float:
        return float(result.get("resolved_width", np.nan))

    def build_one_cell_bank_search_result(
        self,
        existing_result: dict,
        function_used: str,
    ) -> dict:
        self.one_cell_fallback_calls += 1
        return {
            "is_valid": True,
            "i_bank_1_index": 1,
            "i_bank_2_index": 1,
            "resolved_width": self.d_ordinate_dist,
            "reach_top_width_filter_applied": True,
            "reach_top_width_filter_one_cell_fallback_applied": True,
            "function_used": function_used,
        }


def test_width_reconstruction_widens_median_until_banks_are_valid() -> None:
    """A representable wider target should be accepted before fallback."""
    x_section = _WidthReconstructionCrossSection(successful_target=20.0)
    original_result = {
        "is_valid": True,
        "i_bank_1_index": 20,
        "i_bank_2_index": 20,
        "resolved_width": 195.0,
        "reach_top_width_filter_applied": False,
    }

    result = _reconstruct_reach_bank_width_with_fallback(
        x_section,
        original_result,
        median_width=10.0,
        q75=20.0,
    )

    assert x_section.attempted_targets == pytest.approx([10.0, 15.0, 20.0])
    assert x_section.one_cell_fallback_calls == 0
    assert result["resolved_width"] == pytest.approx(20.0)
    assert result["reach_top_width_filter_reconstruction_attempts"] == 3
    assert result["reach_top_width_filter_width_increase_cells"] == 2
    assert result["reach_top_width_filter_post_validation_passed"] is True


def test_width_reconstruction_uses_one_cell_after_ten_failed_increases() -> None:
    """Eleven failed targets should deterministically produce indices (1, 1)."""
    x_section = _WidthReconstructionCrossSection(successful_target=None)
    original_result = {
        "is_valid": True,
        "i_bank_1_index": 20,
        "i_bank_2_index": 20,
        "resolved_width": 195.0,
        "reach_top_width_filter_applied": False,
    }

    result = _reconstruct_reach_bank_width_with_fallback(
        x_section,
        original_result,
        median_width=10.0,
        q75=20.0,
    )

    assert x_section.attempted_targets == pytest.approx(
        [10.0 + 5.0 * index for index in range(11)]
    )
    assert x_section.one_cell_fallback_calls == 1
    assert result["i_bank_1_index"] == 1
    assert result["i_bank_2_index"] == 1
    assert result["reach_top_width_filter_reconstruction_attempts"] == 11
    assert result["reach_top_width_filter_one_cell_fallback_applied"] is True


def test_observation_below_interpolation_becomes_new_downstream_anchor() -> None:
    """A lower observation should refit and extend the approaching slope."""
    surface, outgoing_grades, anchor_mask = (
        _anchor_interpolated_bank_surface_to_cell_observations(
            np.asarray([np.nan, 8.5, np.nan, np.nan]),
            np.asarray([10.0, 9.0, 8.0, 7.0]),
            np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
            300.0,
            7.0,
            0.001,
        )
    )

    assert surface.tolist() == pytest.approx([10.0, 8.5, 7.75, 7.0])
    assert outgoing_grades.tolist() == pytest.approx(
        [0.015, 0.0075, 0.0075, 0.0075]
    )
    assert anchor_mask.tolist() == [False, True, False, False]
    assert np.all(np.diff(surface) < 0.0)


def test_each_lower_observation_refits_from_nearest_upstream_anchor() -> None:
    """Successive low anchors should create continuous piecewise grades."""
    surface, outgoing_grades, anchor_mask = (
        _anchor_interpolated_bank_surface_to_cell_observations(
            np.asarray([np.nan, np.nan, 7.5, 6.4, np.nan]),
            np.asarray([10.0, 9.0, 8.0, 7.0, 6.0]),
            np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
            400.0,
            6.0,
            0.001,
        )
    )

    # The first low observation refits cells 0-2. The second one refits only
    # cells 2-3, and each accepted anchor resets its outgoing grade to finish
    # at the fixed outlet control rather than extrapolating below it.
    assert surface.tolist() == pytest.approx([10.0, 8.75, 7.5, 6.4, 6.0])
    assert outgoing_grades.tolist() == pytest.approx(
        [0.0125, 0.0125, 0.011, 0.004, 0.004]
    )
    assert anchor_mask.tolist() == [False, False, True, True, False]


def test_extreme_low_observation_cannot_pull_surface_below_outlet() -> None:
    """An infeasible low anchor should be clipped to the outlet-grade floor."""
    surface, outgoing_grades, anchor_mask = (
        _anchor_interpolated_bank_surface_to_cell_observations(
            np.asarray([np.nan, -100.0, np.nan]),
            np.asarray([10.0, 9.0, 8.0]),
            np.asarray([0.0, 0.5, 1.0]),
            100.0,
            8.0,
            0.001,
        )
    )

    assert surface.tolist() == pytest.approx([10.0, 8.05, 8.0])
    assert outgoing_grades.tolist() == pytest.approx([0.039, 0.001, 0.001])
    assert anchor_mask.tolist() == [False, True, False]
    assert np.nanmin(surface) == pytest.approx(8.0)


def test_observation_above_interpolation_is_not_used_as_anchor() -> None:
    """An observation above the interpolation should leave it unchanged."""
    surface, _, anchor_mask = (
        _anchor_interpolated_bank_surface_to_cell_observations(
            np.asarray([np.nan, 11.0]),
            np.asarray([10.0, 9.0]),
            np.asarray([0.0, 1.0]),
            100.0,
            9.0,
            0.001,
        )
    )

    assert surface.tolist() == pytest.approx([10.0, 9.0])
    assert anchor_mask.tolist() == [False, False]


def test_observation_anchor_filters_bounds_and_thalweg_values() -> None:
    """Only in-bound observations detectably above the bed may be anchors."""
    surface, _, anchor_mask = (
        _anchor_interpolated_bank_surface_to_cell_observations(
            np.asarray([np.nan, 10.5, 9.5, 8.5, 7.0]),
            np.asarray([12.0, 11.0, 10.0, 9.0, 8.0]),
            np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
            100.0,
            8.0,
            0.001,
            thalweg_elevations=np.asarray([11.0, 9.0, 9.0, 8.5, 6.0]),
            lower_bound=8.0,
            upper_bound=10.0,
        )
    )

    # 10.5 is above the upper bound, 8.5 equals its thalweg, and 7.0 is
    # below the lower bound. Only 9.5 is eligible to reset the interpolation.
    assert surface.tolist() == pytest.approx([12.0, 10.75, 9.5, 8.75, 8.0])
    assert anchor_mask.tolist() == [False, False, True, False, False]
    assert np.all(np.diff(surface) < 0.0)


def test_first_cell_anchor_cannot_exceed_incoming_network_control() -> None:
    """A first anchor must fit between its junction and outlet controls."""
    surface, _, anchor_mask = (
        _anchor_interpolated_bank_surface_to_cell_observations(
            np.asarray([9.0, np.nan]),
            np.asarray([10.0, 9.0]),
            np.asarray([0.0, 1.0]),
            100.0,
            9.0,
            0.001,
            upstream_control_ceiling=10.0,
        )
    )

    # The observed 9.0 m first cell cannot descend to the fixed 9.0 m outlet
    # at the required grade. Raise it only to the feasible 9.1 m floor, which
    # remains below the incoming 10.0 m network control.
    assert surface.tolist() == pytest.approx([9.1, 9.0])
    assert anchor_mask.tolist() == [True, False]


def test_observation_anchor_requires_aligned_arrays() -> None:
    """Observations, interpolation, and fractions must describe the same cells."""
    with pytest.raises(ValueError, match="same shape"):
        _anchor_interpolated_bank_surface_to_cell_observations(
            np.asarray([1.0, 2.0]),
            np.asarray([1.0]),
            np.asarray([0.0]),
            1.0,
            1.0,
        )


def test_network_estimator_stores_observation_anchored_cell_surface() -> None:
    """Production estimator should persist its cell surface on the graph node."""
    graph = nx.DiGraph()
    graph.add_nodes_from([(1, {"length": 100.0}), (2, {"length": 300.0})])
    graph.add_edge(1, 2)

    _estimate_network_smoothed_reach_min_bank_elevations(
        graph,
        {1: 10.0, 2: 7.0},
        {1: 10.0, 2: 10.0},
        {
            2: {
                "ordered_coordinates": np.asarray([0.0, 1.0, 2.0, 3.0]),
                "observed_elevations": np.asarray(
                    [10.0, 8.5, np.nan, 7.0]
                ),
            }
        },
    )

    stored_surface = graph.nodes[2][
        "observed_anchored_cell_bank_elevation_surface"
    ]
    stored_grades = graph.nodes[2]["cell_bank_elevation_outgoing_grades"]
    assert stored_surface.tolist() == pytest.approx([10.0, 8.5, 7.75, 7.0])
    assert np.all(np.diff(stored_surface) <= -MIN_SLOPE * 100.0)
    assert np.all(stored_grades >= MIN_SLOPE)


def test_interpolates_each_cross_section_between_connected_reaches() -> None:
    """Connected reach controls should produce one linear cell surface."""
    graph = nx.DiGraph()
    graph.add_node(10, length=100.0, bank_elevation_grade=0.1)
    graph.add_node(20, length=80.0)
    graph.add_edge(10, 20, length=100.0)

    elevations, fractions, downstream_id, downstream_control = (
        _interpolate_reach_bank_elevation_surface(
            graph,
            10,
            np.asarray([12.0, 7.0, 2.0]),
            {10: 90.0, 20: 80.0},
        )
    )

    assert elevations.tolist() == pytest.approx([100.0, 95.0, 90.0])
    assert fractions.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert downstream_id == 20
    assert downstream_control == pytest.approx(90.0)


def test_outlet_surface_uses_an_assigned_positive_network_grade() -> None:
    """An observed positive grade should descend to the outlet minimum."""
    graph = nx.DiGraph()
    graph.add_node(30, length=200.0, bank_elevation_grade=0.001)

    elevations, fractions, downstream_id, downstream_control = (
        _interpolate_reach_bank_elevation_surface(
            graph,
            30,
            np.asarray([0.0, 1.0, 3.0]),
            {30: 49.8},
        )
    )

    assert fractions.tolist() == pytest.approx([0.0, 1.0 / 3.0, 1.0])
    assert elevations.tolist() == pytest.approx([50.0, 49.9333333333, 49.8])
    assert downstream_id is None
    assert downstream_control == pytest.approx(49.8)


def test_single_cell_reach_places_its_minimum_at_the_outlet_cell() -> None:
    """A one-cell reach should receive its outlet control, not its upstream end."""
    graph = nx.DiGraph()
    graph.add_node(1, length=100.0, bank_elevation_grade=0.01)

    elevations, fractions, _, outlet_control = (
        _interpolate_reach_bank_elevation_surface(
            graph,
            1,
            np.asarray([0.0]),
            {1: 90.0},
        )
    )

    assert fractions.tolist() == pytest.approx([1.0])
    assert elevations.tolist() == pytest.approx([90.0])
    assert outlet_control == pytest.approx(90.0)


def test_degenerate_coordinates_use_the_known_stream_order() -> None:
    """Coincident stations should fall back to uniform ordered fractions."""
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (1, {"length": 10.0, "bank_elevation_grade": 0.2}),
            (2, {"length": 10.0}),
        ]
    )
    graph.add_edge(1, 2, length=10.0)

    elevations, fractions, _, _ = _interpolate_reach_bank_elevation_surface(
        graph,
        1,
        np.asarray([4.0, 4.0, 4.0]),
        {1: 10.0, 2: 8.0},
    )

    assert fractions.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert elevations.tolist() == pytest.approx([12.0, 11.0, 10.0])


def test_surface_uses_zero_grade_when_graph_grade_is_invalid() -> None:
    """An invalid stored grade should become flat rather than rise downstream."""
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (1, {"length": 10.0, "bank_elevation_grade": -1.0}),
            (2, {"length": 10.0}),
        ]
    )
    graph.add_edge(1, 2, length=10.0)

    elevations, _, downstream_id, downstream_control = (
        _interpolate_reach_bank_elevation_surface(
            graph,
            1,
            np.asarray([0.0, 1.0, 2.0]),
            {1: 10.0, 2: 12.0},
        )
    )

    assert elevations.tolist() == pytest.approx([10.0, 10.0, 10.0])
    assert np.all(np.diff(elevations) <= 0.0)
    assert downstream_id == 2
    assert downstream_control == pytest.approx(10.0)


def test_equal_outlet_controls_use_minimum_numerical_grade() -> None:
    """Equal minima should use MIN_SLOPE rather than zero or the old 0.001."""
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (1, {"length": 100.0}),
            (2, {"length": 100.0}),
            (3, {"length": 100.0}),
        ]
    )
    graph.add_edges_from([(1, 2), (2, 3)])

    outlet_controls = _estimate_network_smoothed_reach_min_bank_elevations(
        graph,
        {1: 100.0, 2: 100.0, 3: 100.0},
        {},
    )

    expected_drop = MIN_SLOPE * 100.0
    assert outlet_controls == pytest.approx(
        {
            1: 100.0,
            2: 100.0 - expected_drop,
            3: 100.0 - 2.0 * expected_drop,
        }
    )
    assert graph.nodes[1]["bank_elevation_grade"] == pytest.approx(
        MIN_SLOPE
    )
    assert graph.nodes[2]["bank_elevation_grade"] == pytest.approx(
        MIN_SLOPE
    )
    assert graph.nodes[3]["bank_elevation_grade"] == pytest.approx(
        MIN_SLOPE
    )


def test_terminal_reaches_use_their_available_endpoint_controls() -> None:
    """A headwater may inherit while an outlet uses its filtered minimum."""
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (1, {"length": 100.0}),
            (2, {"length": 100.0}),
            (3, {"length": 100.0}),
        ]
    )
    graph.add_edges_from([(1, 2), (2, 3)])

    outlet_controls = _estimate_network_smoothed_reach_min_bank_elevations(
        graph,
        {1: 100.0, 2: 90.0, 3: 89.0},
        {},
    )

    assert outlet_controls == pytest.approx({1: 100.0, 2: 90.0, 3: 89.0})
    assert graph.nodes[1]["bank_elevation_grade"] == pytest.approx(0.1)
    assert graph.nodes[2]["bank_elevation_grade"] == pytest.approx(0.1)
    assert graph.nodes[3]["bank_elevation_grade"] == pytest.approx(0.01)

    reach_surfaces = {
        reach_id: _interpolate_reach_bank_elevation_surface(
            graph,
            reach_id,
            np.asarray([0.0, 50.0, 100.0]),
            outlet_controls,
        )[0]
        for reach_id in (1, 2, 3)
    }
    assert reach_surfaces[1].tolist() == pytest.approx([110.0, 105.0, 100.0])
    assert reach_surfaces[2].tolist() == pytest.approx([100.0, 95.0, 90.0])
    assert reach_surfaces[3].tolist() == pytest.approx([90.0, 89.5, 89.0])
    assert graph.nodes[3]["outlet_upstream_bank_elevation"] == pytest.approx(
        90.0
    )
    assert graph.nodes[3][
        "outlet_filtered_minimum_bank_elevation"
    ] == pytest.approx(89.0)
    assert graph.nodes[3]["bank_elevation_grade_source"] == (
        "outlet_upstream_minimum_to_filtered_minimum"
    )


def test_outlet_surface_uses_lower_filtered_banks_as_anchors() -> None:
    """Outlet cells below its endpoint line should refit monotonic segments."""
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [(1, {"length": 100.0}), (2, {"length": 100.0})]
    )
    graph.add_edge(1, 2)

    outlet_controls = _estimate_network_smoothed_reach_min_bank_elevations(
        graph,
        {1: 110.0, 2: 100.0},
        {1: 120.0, 2: 110.0},
        {
            2: {
                "ordered_coordinates": np.asarray([0.0, 50.0, 100.0]),
                # The initial outlet line predicts 105 m in the middle. The
                # filtered 103 m bank becomes an anchor before returning to
                # the fixed 100 m outlet minimum.
                "observed_elevations": np.asarray([110.0, 103.0, 100.0]),
                "thalweg_elevations": np.asarray([90.0, 90.0, 90.0]),
                "lower_bound": 100.0,
                "upper_bound": 110.0,
            }
        },
    )

    node_data = graph.nodes[2]
    assert outlet_controls == pytest.approx({1: 110.0, 2: 100.0})
    assert node_data["bank_elevation_grade"] == pytest.approx(0.1)
    assert node_data["baseline_cell_bank_elevation_surface"].tolist() == (
        pytest.approx([110.0, 105.0, 100.0])
    )
    assert node_data[
        "observed_anchored_cell_bank_elevation_surface"
    ].tolist() == pytest.approx([110.0, 103.0, 100.0])
    assert node_data["cell_bank_elevation_outgoing_grades"].tolist() == (
        pytest.approx([0.14, 0.06, 0.06])
    )
    assert node_data[
        "cell_bank_elevation_observation_anchor_mask"
    ].tolist() == [False, True, False]


def test_isolated_reach_uses_maximum_to_minimum_surface_and_anchors() -> None:
    """An isolated stream should receive one explicit downhill workflow."""
    graph = nx.DiGraph()
    graph.add_node(1, length=100.0)

    outlet_controls = _estimate_network_smoothed_reach_min_bank_elevations(
        graph,
        {1: 90.0},
        {1: 100.0},
        {
            1: {
                "ordered_coordinates": np.asarray([0.0, 50.0, 100.0]),
                # The 94 m middle bank is lower than the initial 95 m line and
                # therefore exercises the unchanged shared anchoring pass.
                "observed_elevations": np.asarray([100.0, 94.0, 90.0]),
                "thalweg_elevations": np.asarray([80.0, 80.0, 80.0]),
                "lower_bound": 90.0,
                "upper_bound": 100.0,
            }
        },
    )

    node_data = graph.nodes[1]
    assert outlet_controls == pytest.approx({1: 90.0})
    assert node_data["bank_elevation_grade"] == pytest.approx(0.1)
    assert node_data["isolated_upstream_bank_elevation"] == pytest.approx(
        100.0
    )
    assert node_data["isolated_downstream_bank_elevation"] == pytest.approx(
        90.0
    )
    assert node_data["baseline_cell_bank_elevation_surface"].tolist() == (
        pytest.approx([100.0, 95.0, 90.0])
    )
    assert node_data[
        "observed_anchored_cell_bank_elevation_surface"
    ].tolist() == pytest.approx([100.0, 94.0, 90.0])
    assert node_data["bank_elevation_flow_direction"] == (
        "ordered_filtered_maximum_to_minimum"
    )
    assert node_data["bank_elevation_grade_source"] == (
        "isolated_filtered_maximum_to_minimum"
    )


def test_isolated_reach_order_runs_from_filtered_maximum_to_minimum() -> None:
    """Isolated cell order should follow the downhill raw-bank direction."""
    graph = nx.DiGraph()
    graph.add_node(1, length=2.0)
    reach_entries = [
        {"row": 0, "col": 0},
        {"row": 0, "col": 1},
        {"row": 0, "col": 2},
    ]

    order, stations = _order_reach_stream_cells_from_network(
        graph,
        1,
        reach_entries,
        {1: reach_entries},
        np.asarray([0, 1, 2]),
        np.asarray([90.0, 95.0, 100.0]),
        1.0,
        1.0,
    )

    assert order.tolist() == [2, 1, 0]
    assert stations.tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_headwater_surface_uses_filtered_maximum_and_low_bank_anchors() -> None:
    """A headwater should initialize max-to-min, then honor a lower bank."""
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [(1, {"length": 100.0}), (2, {"length": 100.0})]
    )
    graph.add_edge(1, 2)

    outlet_controls = _estimate_network_smoothed_reach_min_bank_elevations(
        graph,
        {1: 100.0, 2: 95.0},
        {1: 110.0, 2: 95.0},
        {
            1: {
                "ordered_coordinates": np.asarray([0.0, 50.0, 100.0]),
                # The initial 110-to-100 m line predicts 105 m at the middle
                # cell. Its filtered 103 m bank therefore becomes an anchor.
                "observed_elevations": np.asarray([110.0, 103.0, 100.0]),
                "thalweg_elevations": np.asarray([90.0, 90.0, 90.0]),
                "lower_bound": 100.0,
                "upper_bound": 110.0,
                # Production supplies this value directly from the already
                # filtered raw-bank array accumulated for the reach.
                "filtered_maximum_elevation": 110.0,
            }
        },
    )

    node_data = graph.nodes[1]
    assert outlet_controls == pytest.approx({1: 100.0, 2: 95.0})
    assert node_data["bank_elevation_grade"] == pytest.approx(0.1)
    assert node_data["headwater_upstream_bank_elevation"] == pytest.approx(
        110.0
    )
    assert node_data["headwater_outlet_bank_elevation"] == pytest.approx(
        100.0
    )
    assert node_data["baseline_cell_bank_elevation_surface"].tolist() == (
        pytest.approx([110.0, 105.0, 100.0])
    )
    assert node_data[
        "observed_anchored_cell_bank_elevation_surface"
    ].tolist() == pytest.approx([110.0, 103.0, 100.0])
    assert node_data["cell_bank_elevation_outgoing_grades"].tolist() == (
        pytest.approx([0.14, 0.06, 0.06])
    )
    assert node_data[
        "cell_bank_elevation_observation_anchor_mask"
    ].tolist() == [False, True, False]
    assert node_data["bank_elevation_grade_source"] == (
        "headwater_filtered_maximum_to_minimum"
    )


def test_network_orders_a_curved_reach_toward_its_successor() -> None:
    """Raster path ordering should follow a curved reach into its successor."""
    graph = nx.DiGraph()
    graph.add_edge(1, 2, length=4.0)
    reach_entries = [
        {"row": 0, "col": 0},
        {"row": 0, "col": 1},
        {"row": 1, "col": 1},
        {"row": 2, "col": 1},
    ]
    grouped_entries = {
        1: reach_entries,
        2: [{"row": 3, "col": 1}],
    }

    order, stations = _order_reach_stream_cells_from_network(
        graph,
        1,
        reach_entries,
        grouped_entries,
        np.asarray([3, 2, 1, 0]),
        np.asarray([10.0, 10.0, 10.0, 10.0]),
        1.0,
        1.0,
    )

    assert order.tolist() == [0, 1, 2, 3]
    assert stations.tolist() == pytest.approx(
        [0.0, np.sqrt(2.0) - 1.0, np.sqrt(2.0), 1.0 + np.sqrt(2.0)]
    )

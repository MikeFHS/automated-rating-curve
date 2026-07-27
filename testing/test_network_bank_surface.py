from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from arc.Automated_Rating_Curve_Generator import (
    _estimate_network_smoothed_reach_min_bank_elevations,
    _interpolate_reach_bank_elevation_surface,
    _order_reach_stream_cells_from_network,
)


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


def test_equal_outlet_controls_remain_flat_and_monotone() -> None:
    """Equal reach minima must not be given an artificial 0.001 grade."""
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
    )

    assert outlet_controls == pytest.approx({1: 100.0, 2: 100.0, 3: 100.0})
    assert graph.nodes[1]["bank_elevation_grade"] == pytest.approx(0.0)
    assert graph.nodes[2]["bank_elevation_grade"] == pytest.approx(0.0)
    assert graph.nodes[3]["bank_elevation_grade"] == pytest.approx(0.0)


def test_terminal_reaches_inherit_neighbor_slopes() -> None:
    """Headwater and outlet reaches should copy their adjacent interior grade."""
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
    )

    assert outlet_controls == pytest.approx({1: 100.0, 2: 90.0, 3: 80.0})
    assert graph.nodes[1]["bank_elevation_grade"] == pytest.approx(0.1)
    assert graph.nodes[2]["bank_elevation_grade"] == pytest.approx(0.1)
    assert graph.nodes[3]["bank_elevation_grade"] == pytest.approx(0.1)

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
    assert reach_surfaces[3].tolist() == pytest.approx([90.0, 85.0, 80.0])


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

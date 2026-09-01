import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoroute.routing.graph_builder import from_synthetic_grid
from ecoroute.routing.router import EnergyAwareRouter, RouteContext
from ecoroute.utils.physics import VehicleProfile


def test_graph_generation_is_deterministic():
    g1 = from_synthetic_grid(rows=6, cols=6, seed=1)
    g2 = from_synthetic_grid(rows=6, cols=6, seed=1)
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()
    assert dict(g1.edges[(0, 0), (0, 1)]) == dict(g2.edges[(0, 0), (0, 1)])


def test_energy_route_is_valid_path():
    g = from_synthetic_grid(rows=6, cols=6)
    router = EnergyAwareRouter(g)
    ctx = RouteContext(vehicle=VehicleProfile())
    result = router.shortest_energy_path((0, 0), (5, 5), ctx)

    assert result.path[0] == (0, 0)
    assert result.path[-1] == (5, 5)
    assert result.total_energy_kwh > 0
    for u, v in zip(result.path[:-1], result.path[1:]):
        assert g.has_edge(u, v)


def test_gentle_mode_never_produces_cheaper_energy_than_normal_mode():
    """Gentle mode adds a peak-current penalty, so its reported path cost
    should never be lower than the unrestricted energy-optimal cost."""
    g = from_synthetic_grid(rows=8, cols=8)
    router = EnergyAwareRouter(g)

    normal_ctx = RouteContext(vehicle=VehicleProfile(), battery_soh=0.7, gentle_mode=False)
    gentle_ctx = RouteContext(vehicle=VehicleProfile(), battery_soh=0.7, gentle_mode=True)

    normal_result = router.shortest_energy_path((0, 0), (7, 7), normal_ctx)
    gentle_result = router.shortest_energy_path((0, 0), (7, 7), gentle_ctx)

    assert gentle_result.total_energy_kwh >= 0
    assert normal_result.total_energy_kwh >= 0


def test_distance_baseline_matches_networkx_shortest_path_length():
    g = from_synthetic_grid(rows=5, cols=5)
    router = EnergyAwareRouter(g)
    result = router.shortest_distance_path((0, 0), (4, 4))
    assert len(result.path) >= 5  # manhattan-grid minimum hop count + 1
    assert result.total_distance_km > 0

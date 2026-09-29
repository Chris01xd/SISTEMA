"""End-to-end smoke test: runs a short simulation and asserts system invariants."""
from app.simulation.abm.model import HospitalMCEModel
from app.simulation.hybrid_engine import HybridSimulationEngine
from app.simulation.system_dynamics.sd_model import SDParameters


def test_short_simulation_end_to_end():
    abm = HospitalMCEModel(
        n_beds_er=20, n_beds_icu=6, n_beds_ward=30, n_operating_rooms=3,
        n_doctors=10, n_surgeons=4, n_nurses=15, n_ambulances=5,
        severity_distribution={"minor": 0.4, "delayed": 0.3, "immediate": 0.2, "critical": 0.08, "expectant": 0.02},
        expected_victims=80, arrival_curve="surge_decay", arrival_window_minutes=45,
        timestep_minutes=1, seed=1,
    )
    engine = HybridSimulationEngine(abm, SDParameters())
    results = engine.run(n_ticks=90)

    assert len(results) == 90
    for r in results:
        assert 0.0 <= r.saturation_index <= 1.0
        assert r.status_band in {"GREEN", "YELLOW", "RED"}
        assert r.census["beds_occupied"] <= r.census["beds_total"]

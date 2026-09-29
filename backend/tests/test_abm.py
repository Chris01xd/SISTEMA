"""Deterministic-seed tests for ABM agent transitions and invariants."""
from app.simulation.abm.model import HospitalMCEModel


def _make_model(seed: int = 7) -> HospitalMCEModel:
    return HospitalMCEModel(
        n_beds_er=10,
        n_beds_icu=4,
        n_beds_ward=15,
        n_operating_rooms=2,
        n_doctors=5,
        n_surgeons=2,
        n_nurses=8,
        n_ambulances=3,
        severity_distribution={"minor": 0.4, "delayed": 0.3, "immediate": 0.2, "critical": 0.08, "expectant": 0.02},
        expected_victims=50,
        arrival_curve="surge_decay",
        arrival_window_minutes=30,
        timestep_minutes=1,
        seed=seed,
    )


def test_occupied_beds_never_exceed_capacity():
    model = _make_model()
    for _ in range(120):
        model.step()
        census = model.census()
        assert census["beds_occupied"] <= census["beds_total"]


def test_no_negative_stocks():
    model = _make_model()
    for _ in range(120):
        model.step()
        census = model.census()
        assert all(v >= 0 for v in census.values())


def test_patients_are_eventually_processed():
    model = _make_model()
    for _ in range(300):
        model.step()
    census = model.census()
    assert census["total_patients_seen"] > 0
    resolved = census["patients_discharged"] + census["patients_deceased"]
    assert resolved > 0

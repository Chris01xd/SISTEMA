"""Closed-form/invariant checks for the System Dynamics layer."""
from app.simulation.system_dynamics.sd_model import SDParameters, SystemDynamicsModel


def test_saturation_index_bounded_between_0_and_1():
    sd = SystemDynamicsModel(SDParameters())
    sd.sync_from_census(
        {"patients_waiting": 1000, "beds_occupied": 500, "beds_total": 10, "staff_active": 0, "staff_total": 1}
    )
    for _ in range(50):
        sd.step(dt_minutes=1)
    assert 0.0 <= sd.state.saturation_index <= 1.0


def test_saturation_is_zero_when_system_is_empty():
    sd = SystemDynamicsModel(SDParameters())
    sd.sync_from_census(
        {"patients_waiting": 0, "beds_occupied": 0, "beds_total": 50, "staff_active": 0, "staff_total": 10}
    )
    sd.step(dt_minutes=1)
    assert sd.state.saturation_index < 0.05


def test_status_band_thresholds():
    assert SystemDynamicsModel.status_band(0.3) == "GREEN"
    assert SystemDynamicsModel.status_band(0.7) == "YELLOW"
    assert SystemDynamicsModel.status_band(0.95) == "RED"

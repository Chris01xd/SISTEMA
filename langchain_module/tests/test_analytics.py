import json

import pytest

from hospital_assistant import analytics
from hospital_assistant.analytics import QueryError


@pytest.fixture
def data(admissions):
    return analytics.prepare(admissions)


def test_overview_matches_hand_computation(data):
    o = analytics.overview(data)
    assert o["admissions"] == 8 and o["patients"] == 8
    assert o["mean_age"] == 55.6  # (30+45+50+66+70+82+85+17)/8 = 55.625
    assert o["median_los_hours"] == 84.0
    assert o["icu_rate_pct"] == 50.0
    assert o["mortality_rate_pct"] == 37.5
    assert o["severity_available_for_pct"] == 75.0


def test_aggregate_without_grouping(data):
    r = analytics.aggregate(data, "count", filters=[{"column": "sex", "op": "eq", "value": "F"}])
    assert r["value"] == 4 and r["admissions_matched"] == 4


def test_aggregate_grouped_by_sex(data):
    r = analytics.aggregate(data, "mortality_rate_pct", group_by="sex")
    # M: 104, 106, 108 -> 2 of 4 died; F: 107 -> 1 of 4 died.
    assert r["groups"]["M"] == {"value": 50.0, "admissions": 4}
    assert r["groups"]["F"] == {"value": 25.0, "admissions": 4}


def test_grouping_by_age_band_and_hour(data):
    bands = analytics.aggregate(data, "count", group_by="age_band")["groups"]
    assert {k: v["value"] for k, v in bands.items()} == {"0-17": 1, "18-39": 1, "40-64": 2, "65-79": 2, "80+": 2}
    hours = analytics.aggregate(data, "count", group_by="admission_hour")["groups"]
    assert hours["22"]["value"] == 2 and hours["8"]["value"] == 2


def test_unknown_severity_is_its_own_group_not_dropped(data):
    groups = analytics.aggregate(data, "count", group_by="severity_level")["groups"]
    assert sum(g["value"] for g in groups.values()) == 8
    assert sum(1 for k in groups if k in ("<NA>", "nan", "None")) == 1


def test_filters_combine_with_and(data):
    r = analytics.aggregate(
        data, "count",
        filters=[{"column": "age", "op": "gte", "value": 65}, {"column": "diagnosis", "op": "contains", "value": "heart"}],
    )
    assert r["value"] == 3
    r = analytics.aggregate(data, "count", filters=[{"column": "diagnosis", "op": "in", "value": ["Sepsis", "Asthma"]}])
    assert r["value"] == 3


def test_empty_result_returns_none_not_a_crash(data):
    r = analytics.aggregate(data, "mean_age", filters=[{"column": "age", "op": "gt", "value": 200}])
    assert r["value"] is None and r["admissions_matched"] == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda d: analytics.aggregate(d, "median_of_everything"),
        lambda d: analytics.aggregate(d, "count", group_by="patient_id"),
        lambda d: analytics.aggregate(d, "count", filters=[{"column": "patient_id", "op": "eq", "value": 1}]),
        lambda d: analytics.aggregate(d, "count", filters=[{"column": "age", "op": "__class__", "value": 1}]),
        lambda d: analytics.aggregate(d, "count", filters=[{"column": "age", "op": "in", "value": 5}]),
        lambda d: analytics.aggregate(d, "count", filters=[{"column": "age", "op": "gt", "value": "abc"}]),
        lambda d: analytics.top_diagnoses(d, n=0),
        lambda d: analytics.top_diagnoses(d, n=26),
    ],
)
def test_invalid_requests_raise_query_error(data, call):
    with pytest.raises(QueryError):
        call(data)


def test_top_diagnoses(data):
    r = analytics.top_diagnoses(data, n=2)
    assert r["top_diagnoses"] == [
        {"diagnosis": "Heart failure", "admissions": 3},
        {"diagnosis": "Sepsis", "admissions": 2},
    ]


def test_results_never_contain_individual_rows(data):
    payloads = [
        analytics.overview(data),
        analytics.aggregate(data, "count", group_by="sex"),
        analytics.top_diagnoses(data),
    ]
    text = json.dumps(payloads)
    assert "patient_id" not in text and not any(str(pid) in text for pid in range(101, 109))

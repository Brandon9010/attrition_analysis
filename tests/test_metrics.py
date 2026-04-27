import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_stay():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_computes_correct_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "Yes", "Yes", "No"],
        }
    )
    result = attrition_by_department(df)
    sales = result[result["department"] == "Sales"].iloc[0]
    hr = result[result["department"] == "HR"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 2
    assert sales["attrition_rate"] == 100.0
    assert hr["employees"] == 2
    assert hr["leavers"] == 1
    assert hr["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )
    result = attrition_by_department(df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------

def test_attrition_by_overtime_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "overtime": ["Yes", "No"],
            "attrition": ["Yes", "No"],
        }
    )
    result = attrition_by_overtime(df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_computes_correct_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["employees"] == 2
    assert yes_row["leavers"] == 2
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["employees"] == 2
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------

def test_average_income_by_attrition_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "attrition": ["Yes", "No"],
            "monthly_income": [3000, 5000],
        }
    )
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_computes_correct_means():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000, 5000, 7000, 9000],
        }
    )
    result = average_income_by_attrition(df)
    yes_mean = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_mean = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert yes_mean == 4000.0
    assert no_mean == 8000.0


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------

def test_satisfaction_summary_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "job_satisfaction": [1, 2],
            "attrition": ["Yes", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_computes_per_group_rate():
    # Regression test: attrition_rate must be leavers / employees in that group,
    # not leavers / total leavers in the dataset.
    # With this data, sat=1 has 1 leaver out of 2 employees (50%).
    # The old buggy code used total leavers (1) as denominator, giving 100%.
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [1, 1, 2, 2],
            "attrition": ["Yes", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    sat1 = result[result["job_satisfaction"] == 1].iloc[0]
    sat2 = result[result["job_satisfaction"] == 2].iloc[0]
    assert sat1["total_employees"] == 2
    assert sat1["leavers"] == 1
    assert sat1["attrition_rate"] == 50.0
    assert sat2["total_employees"] == 2
    assert sat2["leavers"] == 0
    assert sat2["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_by_satisfaction():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [3, 1, 4, 2],
            "attrition": ["No", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    levels = result["job_satisfaction"].tolist()
    assert levels == sorted(levels)

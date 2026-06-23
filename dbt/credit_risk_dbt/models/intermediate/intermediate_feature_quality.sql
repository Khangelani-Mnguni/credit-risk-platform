{{ config(
    materialized='table',
    schema='intermediate',
    alias='intermediate_feature_quality'
) }}

select

    count(*) as total_records,

    countif(annual_inc is null)
        as missing_annual_income,

    countif(fico_score is null)
        as missing_fico_score,

    countif(revol_util_pct is null)
        as missing_revol_utilization,

    countif(dti is null)
        as missing_dti,

    countif(credit_history_months is null)
        as missing_credit_history,

    countif(loan_to_income_ratio is null)
        as missing_loan_income_ratio,

    countif(fico_change is null)
        as missing_fico_change,

    avg(fico_score)
        as avg_fico_score,

    avg(annual_inc)
        as avg_annual_income,

    avg(revol_util_pct)
        as avg_revol_utilization,

    avg(dti)
        as avg_dti

from {{ ref('intermediate_credit_risk_features') }}
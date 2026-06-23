{{ config(
    materialized='table',
    schema='intermediate',
    alias='intermediate_credit_risk_features'
) }}

with features as (

select

    lc.id,

    -- LOAN

    lc.loan_amnt,
    lc.funded_amnt,
    lc.funded_amnt_inv,
    lc.int_rate_pct,
    lc.installment,
    lc.grade,
    lc.sub_grade,
    lc.purpose,
    lc.application_type,

    -- BORROWER

    b.emp_length_years,
    b.annual_inc,
    b.home_ownership,
    b.dti,

    -- CREDIT

    ch.fico_score,
    ch.last_fico_score,

    -- DELINQUENCY

    d.delinq_2yrs,
    d.acc_now_delinq,
    d.inq_last_6mths,
    d.inq_last_12m,

    -- ACCOUNTS

    a.open_acc,
    a.total_acc,
    a.mort_acc,
    a.pub_rec,

    -- BALANCES

    bal.revol_bal,
    bal.revol_util_pct,
    bal.total_rev_hi_lim,
    bal.tot_cur_bal,
    bal.total_bc_limit,
    bal.bc_util,
    bal.all_util,

    -- DATES

    dt.issue_date,
    dt.earliest_credit_line_date,

    -- =====================================================
    -- ENGINEERED FEATURES
    -- =====================================================

    date_diff(
        dt.issue_date,
        dt.earliest_credit_line_date,
        month
    ) as credit_history_months,

    safe_divide(
        lc.loan_amnt,
        b.annual_inc
    ) as loan_to_income_ratio,

    safe_divide(
        lc.installment,
        b.annual_inc
    ) as installment_income_ratio,

    safe_divide(
        lc.funded_amnt,
        b.annual_inc
    ) as funded_to_income_ratio,

    ch.last_fico_score
        - ch.fico_score
        as fico_change,

    case
        when d.delinq_2yrs > 0 then 1
        else 0
    end as delinquency_flag,

    case
        when d.inq_last_6mths > 0 then 1
        else 0
    end as recent_inquiry_flag,

    case
        when bal.revol_util_pct > 80 then 1
        else 0
    end as high_utilization_flag,

    case
        when a.total_acc < 5 then 1
        else 0
    end as thin_file_flag,

    case
        when a.mort_acc > 0 then 1
        else 0
    end as mortgage_flag,

    case
        when b.emp_length_years >= 5 then 1
        else 0
    end as employment_stability_flag,

    case
        when ch.fico_score < 580 then 'Poor'
        when ch.fico_score < 670 then 'Fair'
        when ch.fico_score < 740 then 'Good'
        when ch.fico_score < 800 then 'Very Good'
        else 'Exceptional'
    end as fico_band,

    case
        when b.dti > 40 then 1
        else 0
    end as high_dti_flag,

    safe_divide(
        bal.revol_bal,
        bal.total_rev_hi_lim
    ) as revolving_balance_ratio,

    safe_divide(
        bal.tot_cur_bal,
        bal.total_bc_limit
    ) as balance_to_limit_ratio,

    case
        when ch.fico_score < 660 then 1
        else 0
    end as subprime_flag

from {{ ref('staging_loan_core') }} lc

left join {{ ref('staging_borrower') }} b
    on lc.id = b.id

left join {{ ref('staging_credit_history') }} ch
    on lc.id = ch.id

left join {{ ref('staging_delinquency') }} d
    on lc.id = d.id

left join {{ ref('staging_accounts') }} a
    on lc.id = a.id

left join {{ ref('staging_balances') }} bal
    on lc.id = bal.id

left join {{ ref('staging_dates') }} dt
    on lc.id = dt.id

)

select *
from features
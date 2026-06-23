{{ config(
    materialized='view',
    alias='staging_dates'
) }}

select

    id,

    {{ parse_month_year('issue_d') }}
        as issue_date,

    {{ parse_month_year('earliest_cr_line') }}
        as earliest_credit_line_date,

    {{ parse_month_year('last_pymnt_d') }}
        as last_payment_date,

    {{ parse_month_year('next_pymnt_d') }}
        as next_payment_date,

    {{ parse_month_year('last_credit_pull_d') }}
        as last_credit_pull_date,

    {{ parse_month_year('sec_app_earliest_cr_line') }}
        as sec_app_earliest_credit_line_date,

    {{ parse_month_year('hardship_start_date') }}
        as hardship_start_date,

    {{ parse_month_year('hardship_end_date') }}
        as hardship_end_date,

    {{ parse_month_year('payment_plan_start_date') }}
        as payment_plan_start_date

from {{ source('raw_tables','loan_status_2007_2023') }}
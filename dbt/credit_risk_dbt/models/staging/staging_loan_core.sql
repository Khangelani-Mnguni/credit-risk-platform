{{ config(
    materialized='view',
    alias='staging_loan_core'
) }}

select

    id,

    loan_amnt,
    funded_amnt,
    funded_amnt_inv,

    term,

    {{ parse_percentage('int_rate') }}
        as int_rate_pct,

    installment,

    grade,
    sub_grade,

    loan_status,

    purpose,
    title,

    pymnt_plan,

    application_type,
    initial_list_status,

    policy_code

from {{ source('raw_tables','loan_status_2007_2023') }}
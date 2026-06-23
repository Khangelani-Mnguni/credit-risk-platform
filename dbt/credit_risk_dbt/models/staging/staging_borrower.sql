{{ config(
    materialized='view',
    alias='staging_borrower'
) }}

select

    id,

    emp_title,

    emp_length,

    {{ emp_length_to_years('emp_length') }}
        as emp_length_years,

    annual_inc,

    verification_status,

    home_ownership,

    zip_code,

    addr_state,

    dti

from {{ source('raw_tables','loan_status_2007_2023') }}
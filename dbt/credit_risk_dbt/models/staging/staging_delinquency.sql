{{ config(
    materialized='view',
    alias='staging_delinquency'
) }}

select

    id,

    delinq_2yrs,
    mths_since_last_delinq,
    mths_since_last_record,
    mths_since_last_major_derog,

    acc_now_delinq,
    delinq_amnt,

    inq_last_6mths,
    inq_fi,
    inq_last_12m

from {{ source('raw_tables','loan_status_2007_2023') }}
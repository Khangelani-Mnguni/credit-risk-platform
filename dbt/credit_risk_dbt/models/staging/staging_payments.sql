{{ config(
    materialized='view',
    alias='staging_payments'
) }}

select

    id,

    out_prncp,
    out_prncp_inv,

    total_pymnt,
    total_pymnt_inv,

    total_rec_prncp,
    total_rec_int,
    total_rec_late_fee,

    recoveries,

    collection_recovery_fee,

    last_pymnt_amnt

from {{ source('raw_tables','loan_status_2007_2023') }}
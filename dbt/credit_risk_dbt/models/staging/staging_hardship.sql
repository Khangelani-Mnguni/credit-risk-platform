{{ config(
    materialized='view',
    alias='staging_hardship'
) }}

select
id,
hardship_flag,
hardship_type,
hardship_reason,
hardship_status,
deferral_term,
hardship_amount,
hardship_length,
hardship_dpd,
hardship_loan_status,
orig_projected_additional_accrued_interest,
hardship_payoff_balance_amount,
hardship_last_payment_amount,
debt_settlement_flag,

from {{ source('raw_tables','loan_status_2007_2023') }}
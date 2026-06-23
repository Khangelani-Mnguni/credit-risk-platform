{{ config(
    materialized='view',
    alias='staging_balances'
) }}

select

    id,

    revol_bal,

    {{ parse_percentage('revol_util') }}
        as revol_util_pct,

    total_rev_hi_lim,

    tot_coll_amt,
    tot_cur_bal,
    avg_cur_bal,

    tot_hi_cred_lim,

    total_bal_ex_mort,

    total_bc_limit,

    total_il_high_credit_limit,

    bc_open_to_buy,
    bc_util,

    max_bal_bc,

    all_util

from {{ source('raw_tables','loan_status_2007_2023') }}
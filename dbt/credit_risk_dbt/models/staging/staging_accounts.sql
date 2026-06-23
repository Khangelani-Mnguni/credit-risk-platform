{{ config(
    materialized='view',
    alias='staging_accounts'
) }}

select

    id,

    open_acc,
    pub_rec,
    total_acc,

    mort_acc,

    num_accts_ever_120_pd,
    num_actv_bc_tl,
    num_actv_rev_tl,
    num_bc_sats,
    num_bc_tl,
    num_il_tl,
    num_op_rev_tl,
    num_rev_accts,
    num_rev_tl_bal_gt_0,
    num_sats,
    num_tl_120dpd_2m,
    num_tl_30dpd,
    num_tl_90g_dpd_24m,
    num_tl_op_past_12m

from {{ source('raw_tables','loan_status_2007_2023') }}
{{ config(
    materialized='view',
    alias='staging_secondary_applicant'
) }}

select
id,
revol_bal_joint,
sec_app_fico_range_low,
sec_app_fico_range_high,
sec_app_earliest_cr_line,
sec_app_inq_last_6mths,
sec_app_mort_acc,
sec_app_open_acc,
sec_app_revol_util,
sec_app_open_act_il,
sec_app_num_rev_accts,
sec_app_chargeoff_within_12_mths,
sec_app_collections_12_mths_ex_med,
    {{ fico_midpoint(
        'sec_app_fico_range_low',
        'sec_app_fico_range_high'
    ) }}
        as sec_app_fico_score

from {{ source('raw_tables','loan_status_2007_2023') }}
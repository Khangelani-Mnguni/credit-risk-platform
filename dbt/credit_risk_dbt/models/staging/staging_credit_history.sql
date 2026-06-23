{{ config(
    materialized='view',
    alias='staging_credit_history'
) }}

select

    id,

    fico_range_low,
    fico_range_high,

    {{ fico_midpoint(
        'fico_range_low',
        'fico_range_high'
    ) }}
        as fico_score,

    last_fico_range_low,
    last_fico_range_high,

    {{ fico_midpoint(
        'last_fico_range_low',
        'last_fico_range_high'
    ) }}
        as last_fico_score

from {{ source('raw_tables','loan_status_2007_2023') }}
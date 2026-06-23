{{ config(
    materialized='table',
    schema='intermediate',
    alias='intermediate_default_target'
) }}

select

    id,

    loan_status,

    case
        when loan_status in (
            'Charged Off',
            'Default',
            'Late (31-120 days)',
            'Does not meet the credit policy. Status:Charged Off'
        )
        then 1
        else 0
    end as default_flag

from {{ ref('staging_loan_core') }}
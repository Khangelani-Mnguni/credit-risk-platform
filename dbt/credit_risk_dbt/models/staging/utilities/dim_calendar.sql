{{ config(materialized='table') }}

with date_spine as (

    {{
        dbt_utils.date_spine(
            datepart="day",
            start_date="cast('2007-01-01' as date)",
            end_date="cast('2030-12-31' as date)"
        )
    }}

)

select
    cast(date_day as date) as calendar_date,
    extract(year from date_day) as year,
    extract(month from date_day) as month,
    extract(quarter from date_day) as quarter,
    format_date('%Y-%m', date_day) as year_month,
    extract(dayofweek from date_day) as day_of_week

from date_spine
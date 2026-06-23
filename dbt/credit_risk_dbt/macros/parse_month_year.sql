{% macro parse_month_year(column_name) %}

safe.parse_date(
    '%b-%Y',
    {{ column_name }}
)

{% endmacro %}
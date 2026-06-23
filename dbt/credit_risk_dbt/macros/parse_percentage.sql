{% macro parse_percentage(column_name) %}

safe_cast(
    replace({{ column_name }}, '%', '')
    as numeric
)

{% endmacro %}
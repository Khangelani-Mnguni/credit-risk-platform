{% macro fico_midpoint(low_col, high_col) %}

(
    {{ low_col }}
    +
    {{ high_col }}
) / 2

{% endmacro %}
{% macro emp_length_to_years(column_name) %}

case

    when {{ column_name }} = '< 1 year'
        then 0

    when {{ column_name }} = '10+ years'
        then 10

    else safe_cast(
        regexp_extract(
            {{ column_name }},
            r'\d+'
        )
        as int64
    )

end

{% endmacro %}
{{ config(materialized='view') }}

select
    courseid as course_id,
    course_title,
    course_url,
    course_mapping_keywords,
    topic_id
from {{ ref('unified_courses_v1') }}

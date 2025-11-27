{{ config(materialized='view') }}

with base as (
    select
        course_id,
        course_title,
        course_url,
        course_mapping_keywords,
        topic_id
    from {{ ref('stg_unified_courses') }}
),

enhanced as (
    select
        *,
        length(course_title) as title_length,
        case 
            when topic_id is null then 'unknown'
            else topic_id
        end as topic_clean
    from base
)

select * from enhanced

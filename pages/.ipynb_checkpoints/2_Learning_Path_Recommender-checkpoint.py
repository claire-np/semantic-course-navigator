import os
import sys
import streamlit as st

# Path Setup
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "app"))

from app.utils_learning_path import generate_learning_path

# CUSTOM CSS

CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top left, #1f2933 0, #020617 55%, #000000 100%);
    color: #e5e7eb;
    font-family: "Inter", sans-serif;
}

/* ---------- Titles ---------- */
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 0.95rem;
    color: #9ca3af;
    max-width: 900px;
    margin-bottom: 1.4rem;
}

/* ---------- Stage Title ---------- */
.stage-title {
    font-size: 1.7rem;
    font-weight: 700;
    margin: 2.0rem 0 0.6rem 0;
}

/* ---------- Zigzag Wrapper & Rows ---------- */
.skill-row-wrap {
    display: flex;
    width: 100%;
    align-items: stretch;    /* ensures equal height columns */
}

.skill-row {
    margin-bottom: 1.8rem;
}

.skill-row-group {
    margin-bottom: 3.2rem;   /* larger gap between row groups */
}

/* ---------- Columns ---------- */
.skill-col {
    width: 50%;
    position: relative;
    padding-left: 1.6rem;
}

.col-separator {
    width: 1px;
    background: rgba(148, 163, 184, 0.08);
}

/* ---------- Skill badge ---------- */
.skill-badge {
    display: inline-block;
    font-size: 0.80rem;
    font-weight: 600;
    padding: 6px 13px;
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    margin-bottom: 0.55rem;
}

/* ---------- Timeline Dot ---------- */
.skill-dot {
    position: absolute;
    left: 0;
    top: 5px;
    width: 11px;
    height: 11px;
    background: #60a5fa;
    border-radius: 999px;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.35);
}

/* ---------- Course card ---------- */
.course-card {
    padding: 0.45rem 0 0.35rem 0;
    margin-bottom: 0.55rem;
    border-bottom: 1px solid rgba(31,41,55,0.85);
}
.course-title a {
    font-size: 0.97rem;
    font-weight: 600;
    color: #e5e7eb !important;
    text-decoration: none;
}
.course-title a:hover {
    color: #93c5fd !important;
}
.meta-row {
    font-size: 0.78rem;
    color: #9ca3af;
}

/* rating + enrollment badges */
.rating-badge,
.enroll-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    margin-right: 0.45rem;
}
.rating-badge { background: rgba(250,204,21,0.10); color: #facc15; }
.enroll-badge { background: rgba(52,211,153,0.10); color: #6ee7b7; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



# SIDEBAR

ROLE_OPTIONS = {
    "Data Analyst": "data_analyst",
    "BI Analyst": "bi_analyst",
    "Data Engineer": "data_engineer",
    "Data Scientist": "data_scientist",
    "AI Engineer": "ai_engineer",
    "DevOps Engineer": "devops",
    "Full-stack Developer": "full_stack",
    "Cybersecurity Engineer": "cybersecurity",
    "Software Architect": "software_architect",
}

ROLE_LABELS = list(ROLE_OPTIONS.keys())

with st.sidebar:
    st.markdown("### 📚 Career Learning Path")
    st.caption("Select a role to see your personalized roadmap.")
    selected_role_label = st.radio("", ROLE_LABELS, index=0)
    st.caption(f"Current role: **{selected_role_label}**")
    show_raw_urls = st.checkbox("Show raw Udemy URLs", value=False)



# HEADER

st.markdown(
    f'<div class="main-title">Career Learning Path – {selected_role_label}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="sub-title">
    A skill-based roadmap designed with curated Udemy courses across each stage.
    </div>
    """,
    unsafe_allow_html=True,
)

role_key = ROLE_OPTIONS[selected_role_label]

with st.spinner("Building your learning path..."):
    lp = generate_learning_path(role_key)

if "error" in lp:
    st.error(lp["error"])
    st.stop()



# STAGE RENDER

STAGE_ORDER = [
    "Stage 1 – Foundations",
    "Stage 2 – Core Skills",
    "Stage 3 – Specialization",
    "Stage 4 – Portfolio & Certification",
]

for stage_name in STAGE_ORDER:
    if stage_name not in lp:
        continue

    stage_blocks = lp[stage_name]

    # Stage title
    st.markdown(f'<div class="stage-title">{stage_name}</div>', unsafe_allow_html=True)

    # Pair blocks in rows of 2
    rows = []
    temp = []
    for block in stage_blocks:
        temp.append(block)
        if len(temp) == 2:
            rows.append(temp)
            temp = []
    if temp:
        rows.append(temp)

    # Render each paired row
    for idx, row in enumerate(rows):
        # Decide spacing class
        row_class = "skill-row-group" if idx == 1 else "skill-row"

        st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)
        st.markdown('<div class="skill-row-wrap">', unsafe_allow_html=True)

        col_left, col_sep, col_right = st.columns([1, 0.03, 1])

        # LEFT COLUMN
        with col_left:
            block = row[0]
            skill = block["skill"]
            courses = block["courses"]

            st.markdown(
                f'<div class="skill-dot"></div><div class="skill-badge">{skill}</div>',
                unsafe_allow_html=True,
            )

            for c in courses:
                title = c.get("course_title", "")
                instructor = c.get("instructor_title", "")
                category = c.get("category", "")
                subcategory = c.get("subcategory", "")
                rating = c.get("avg_rating_90d")
                enroll = c.get("enrollments")
                url = c.get("course_url")

                st.markdown('<div class="course-card">', unsafe_allow_html=True)

                if url:
                    st.markdown(
                        f'<div class="course-title"><a href="{url}" target="_blank">{title}</a></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f'<div class="course-title">{title}</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div class="meta-row">By <strong>{instructor}</strong> · '
                    f'{category} → {subcategory}</div>',
                    unsafe_allow_html=True,
                )

                rating_html = f'<span class="rating-badge">⭐ {rating:.2f}</span>' if rating else ""
                enroll_html = f'<span class="enroll-badge">👥 {enroll:,}</span>' if enroll else ""

                st.markdown(
                    f'<div style="margin-top:4px;">{rating_html}{enroll_html}</div>',
                    unsafe_allow_html=True,
                )

                if show_raw_urls and url:
                    st.caption(url)

                st.markdown("</div>", unsafe_allow_html=True)

        # SEPARATOR
        with col_sep:
            st.markdown('<div class="col-separator" style="height:100%;"></div>', unsafe_allow_html=True)

        # RIGHT COLUMN (if exists)
        if len(row) == 2:
            with col_right:
                block = row[1]
                skill = block["skill"]
                courses = block["courses"]

                st.markdown(
                    f'<div class="skill-dot"></div><div class="skill-badge">{skill}</div>',
                    unsafe_allow_html=True,
                )

                for c in courses:
                    title = c.get("course_title", "")
                    instructor = c.get("instructor_title", "")
                    category = c.get("category", "")
                    subcategory = c.get("subcategory", "")
                    rating = c.get("avg_rating_90d")
                    enroll = c.get("enrollments")
                    url = c.get("course_url")

                    st.markdown('<div class="course-card">', unsafe_allow_html=True)

                    if url:
                        st.markdown(
                            f'<div class="course-title"><a href="{url}" target="_blank">{title}</a></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f'<div class="course-title">{title}</div>', unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="meta-row">By <strong>{instructor}</strong> · '
                        f'{category} → {subcategory}</div>',
                        unsafe_allow_html=True,
                    )

                    rating_html = f'<span class="rating-badge">⭐ {rating:.2f}</span>' if rating else ""
                    enroll_html = f'<span class="enroll-badge">👥 {enroll:,}</span>' if enroll else ""

                    st.markdown(
                        f'<div style="margin-top:4px;">{rating_html}{enroll_html}</div>',
                        unsafe_allow_html=True,
                    )

                    if show_raw_urls and url:
                        st.caption(url)

                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # end skill-row-wrap
        st.markdown("</div>", unsafe_allow_html=True)  # end row class

    st.write("")

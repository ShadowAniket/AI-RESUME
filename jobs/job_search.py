import streamlit as st
from typing import List, Dict
from .job_portals import JobPortal
from .suggestions import (
    JOB_SUGGESTIONS, 
    LOCATION_SUGGESTIONS, 
    EXPERIENCE_RANGES,
    SALARY_RANGES,
    JOB_TYPES
)
from .companies import get_featured_companies, get_market_insights

def filter_suggestions(query: str, suggestions: List[Dict]) -> List[Dict]:
    """Filter suggestions based on user input"""
    if not query:
        return []
    return [
        s for s in suggestions 
        if query.lower() in s["text"].lower()
    ][:5]

def get_filter_options():
    """Get filter options for job search"""
    return {
        "experience_levels": [
            {"id": "all", "text": "All Levels"},
            {"id": "0-1", "text": "0-1 years"},
            {"id": "1-3", "text": "1-3 years"},
            {"id": "3-5", "text": "3-5 years"},
            {"id": "5-7", "text": "5-7 years"},
            {"id": "7-10", "text": "7-10 years"},
            {"id": "10+", "text": "10+ years"}
        ],
        "salary_ranges": [
            {"id": "all", "text": "All Ranges"},
            {"id": "0-3", "text": "0-3 LPA"},
            {"id": "3-6", "text": "3-6 LPA"},
            {"id": "6-10", "text": "6-10 LPA"},
            {"id": "10-15", "text": "10-15 LPA"},
            {"id": "15+", "text": "15+ LPA"}
        ],
        "job_types": [
            {"id": "all", "text": "All Types"},
            {"id": "full-time", "text": "Full Time"},
            {"id": "part-time", "text": "Part Time"},
            {"id": "contract", "text": "Contract"},
            {"id": "remote", "text": "Remote"}
        ]
    }

def render_company_section():
    """Render the featured companies section"""
    st.markdown("""
        <style>
        .company-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
            padding: 1rem 0;
        }
        .company-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            transition: transform 0.2s;
            cursor: pointer;
        }
        .company-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
        }
        .company-header {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .company-icon {
            font-size: 1.5rem;
            margin-right: 0.5rem;
        }
        .company-categories {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .company-category {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Featured Companies
    st.markdown("### 🏢 Featured Companies")
    
    tabs = st.tabs(["All Companies", "Tech Giants", "Indian Tech", "Global Corps"])
    
    categories = [None, "tech", "indian_tech", "global_corps"]
    for tab, category in zip(tabs, categories):
        with tab:
            companies = get_featured_companies(category)
            st.markdown('<div class="company-grid">', unsafe_allow_html=True)
            
            for company in companies:
                st.markdown(f"""
                    <a href="{company['careers_url']}" target="_blank" style="text-decoration: none; color: inherit;">
                        <div class="company-card">
                            <div class="company-header">
                                <i class="{company['icon']} company-icon" style="color: {company['color']}"></i>
                                <h3 style="margin: 0;">{company['name']}</h3>
                            </div>
                            <p style="margin: 0.5rem 0; color: #888;">{company['description']}</p>
                            <div class="company-categories">
                                {' '.join(f'<span class="company-category">{cat}</span>' for cat in company['categories'])}
                            </div>
                        </div>
                    </a>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_market_insights():
    """Render job market insights section"""
    insights = get_market_insights()
    
    st.markdown("""
        <style>
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            padding: 1rem 0;
        }
        .insight-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            transition: transform 0.3s ease, background 0.3s ease;
        }
        .insight-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
        }
        .insight-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: #00bfa5;
        }
        .growth-text {
            color: #00c853;
            font-weight: bold;
        }
        .salary-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            border-left: 4px solid #00bfa5;
        }
        .salary-card:hover {
            transform: translateX(10px);
            background: rgba(255, 255, 255, 0.08);
        }
        .salary-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .role-icon {
            font-size: 1.5rem;
            margin-right: 1rem;
            color: #00bfa5;
        }
        .salary-details {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.5rem;
        }
        .salary-tag {
            background: rgba(0, 191, 165, 0.1);
            color: #00bfa5;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .experience-tag {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .role-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin: 0;
        }
        .salary-range {
            font-size: 1.1rem;
            color: #00bfa5;
            font-weight: bold;
        }
        .role-icons {
            font-family: "Font Awesome 5 Free";
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Job Market Insights")
    
    tabs = st.tabs(["Trending Skills", "Top Locations", "Salary Insights"])
    
    with tabs[0]:
        st.markdown('<div class="insights-grid">', unsafe_allow_html=True)
        for skill in insights["trending_skills"]:
            st.markdown(f"""
                <div class="insight-card">
                    <i class="{skill['icon']} insight-icon"></i>
                    <h4>{skill['name']}</h4>
                    <p class="growth-text">Growth: {skill['growth']}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown('<div class="insights-grid">', unsafe_allow_html=True)
        for location in insights["top_locations"]:
            st.markdown(f"""
                <div class="insight-card">
                    <i class="{location['icon']} insight-icon"></i>
                    <h4>{location['name']}</h4>
                    <p>Available Jobs: {location['jobs']}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        # Role-specific icons
        role_icons = {
            "Software Engineer": "fas fa-code",
            "Data Scientist": "fas fa-brain",
            "Product Manager": "fas fa-tasks",
            "DevOps Engineer": "fas fa-server",
            "UI/UX Designer": "fas fa-paint-brush"
        }
        
        for insight in insights["salary_insights"]:
            role = insight['role']
            icon = role_icons.get(role, "fas fa-briefcase")
            
            st.markdown(f"""
                <div class="salary-card">
                    <div class="salary-header">
                        <i class="{icon} role-icon"></i>
                        <div>
                            <h3 class="role-title">{role}</h3>
                            <div class="salary-details">
                                <span class="salary-tag">₹ {insight['range']}</span>
                                <span class="experience-tag">
                                    <i class="fas fa-history"></i> {insight['experience']}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_job_search():
    """Render job search page with enhanced features"""
    st.title("🔍 Smart Job Search")
    st.markdown("Find Your Dream Job Across Multiple Platforms")
    
    # Market Insights Section (Above Search)
    render_market_insights()
    
    # Job Search Section
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        # Search inputs
        col1, col2 = st.columns([2, 1])
        
        with col1:
            job_query = st.text_input("Job Title / Skills", 
                                    value="", 
                                    placeholder="e.g. Software Engineer, Data Scientist")
            
            if job_query and len(job_query) >= 2:
                filtered_jobs = [s["text"] for s in JOB_SUGGESTIONS if job_query.lower() in s["text"].lower()]
                if filtered_jobs:
                    job_query = st.selectbox("Select Job Title", filtered_jobs)
        
        with col2:
            location = st.text_input("Location", 
                                   value="",
                                   placeholder="e.g. Bangalore, Mumbai")
            
            if location and len(location) >= 2:
                filtered_locations = [s["text"] for s in LOCATION_SUGGESTIONS if location.lower() in s["text"].lower()]
                if filtered_locations:
                    location = st.selectbox("Select Location", filtered_locations)

        # Advanced Filters
        with st.expander("🎯 Advanced Filters"):
            st.markdown('<div class="filter-section">', unsafe_allow_html=True)
            filter_cols = st.columns(3)
            
            with filter_cols[0]:
                experience = st.selectbox("Experience Level",
                                        options=get_filter_options()["experience_levels"],
                                        format_func=lambda x: x["text"])
            
            with filter_cols[1]:
                salary_range = st.selectbox("Salary Range",
                                          options=get_filter_options()["salary_ranges"],
                                          format_func=lambda x: x["text"])
            
            with filter_cols[2]:
                job_type = st.selectbox("Job Type",
                                      options=get_filter_options()["job_types"],
                                      format_func=lambda x: x["text"])
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Search button
        if st.button("SEARCH JOBS", type="primary"):
            if job_query:
                job_portal = JobPortal()
                results = job_portal.search_jobs(job_query, location, experience)
                
                if results:
                    st.markdown("### 🎯 Job Search Results")
                    for result in results:
                        with st.container():
                            st.markdown(f"""
                            <div style='padding: 10px; margin: 5px 0; border-radius: 5px; background: rgba(255,255,255,0.05);'>
                                <h4>
                                    <i class='{result["icon"]}' style='color: {result["color"]}'></i>
                                    {result["portal"]}
                                </h4>
                                <p>{result["title"]}</p>
                                <a href='{result["url"]}' target='_blank' style='color: #00bfa5;'>
                                    View Jobs on {result["portal"]} →
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No results found. Try different search terms or filters.")
            else:
                st.warning("Please enter a job title or skills to search.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Featured Companies Section (Below Search)
    render_company_section()

# Removed render_job_search() call to prevent automatic rendering

import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="Smart Resume AI",
    page_icon="🚀",
    layout="wide"
)

import json
import pandas as pd
import plotly.express as px
import traceback
from utils.resume_analyzer import ResumeAnalyzer
from utils.resume_builder import ResumeBuilder
from config.database import (
    get_database_connection, save_resume_data, save_analysis_data, 
    init_database, verify_admin, log_admin_action
)
from config.job_roles import JOB_ROLES
from config.courses import COURSES_BY_CATEGORY, RESUME_VIDEOS, INTERVIEW_VIDEOS, get_courses_for_role, get_category_for_role
from dashboard.dashboard import DashboardManager
import requests
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
import base64
import io
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from feedback.feedback import FeedbackManager
from ui_components import (
    apply_modern_styles, hero_section, feature_card, about_section, 
    page_header, render_analytics_section, render_activity_section, 
    render_suggestions_section
)
from datetime import datetime
from jobs.job_search import render_job_search
from PIL import Image

class ResumeApp:
    def __init__(self):
        """Initialize the application"""
        if 'form_data' not in st.session_state:
            st.session_state.form_data = {
                'personal_info': {
                    'full_name': '',
                    'email': '',
                    'phone': '',
                    'location': '',
                    'linkedin': '',
                    'portfolio': ''
                },
                'summary': '',
                'experiences': [],
                'education': [],
                'projects': [],
                'skills_categories': {
                    'technical': [],
                    'soft': [],
                    'languages': [],
                    'tools': []
                }
            }
        
        # Initialize navigation state
        if 'page' not in st.session_state:
            st.session_state.page = 'home'
            
        # Initialize admin state
        if 'is_admin' not in st.session_state:
            st.session_state.is_admin = False
        
        self.pages = {
            "🏠 HOME": self.render_home,
            "🔍 RESUME ANALYZER": self.render_analyzer,
            "📝 RESUME BUILDER": self.render_builder,
            "📊 DASHBOARD": self.render_dashboard,
            "🎯 JOB SEARCH": self.render_job_search,
            "💬 FEEDBACK": self.render_feedback_page,
            "ℹ️ ABOUT": self.render_about
        }
        
        # Initialize dashboard manager
        self.dashboard_manager = DashboardManager()
        
        self.analyzer = ResumeAnalyzer()
        self.builder = ResumeBuilder()
        self.job_roles = JOB_ROLES
        
        # Initialize session state
        if 'user_id' not in st.session_state:
            st.session_state.user_id = 'default_user'
        if 'selected_role' not in st.session_state:
            st.session_state.selected_role = None
        
        # Initialize database
        init_database()
        
        # Load external CSS
        with open('style/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
        # Load Google Fonts
        st.markdown("""
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        """, unsafe_allow_html=True)

    def load_lottie_url(self, url: str):
        """Load Lottie animation from URL with fallback"""
        try:
            r = requests.get(url, timeout=5)  # Add timeout
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        
        # Fallback: return a simple loading animation
        return {
            "v": "5.7.4",
            "fr": 60,
            "ip": 0,
            "op": 120,
            "w": 200,
            "h": 200,
            "assets": [],
            "layers": [{
                "ddd": 0,
                "ind": 1,
                "ty": 4,
                "nm": "Circle",
                "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {
                        "a": 1,
                        "k": [{"t": 0, "s": [0]}, {"t": 120, "s": [360]}],
                        "ix": 10
                    },
                    "p": {"a": 0, "k": [100, 100, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {"a": 0, "k": [100, 100, 100]}
                },
                "shapes": [{
                    "ty": "el",
                    "p": {"a": 0, "k": [0, 0]},
                    "s": {"a": 0, "k": [60, 60]},
                    "c": {"a": 0, "k": [0, 0.6, 1]}
                }]
            }]
        }

    def apply_global_styles(self):
        # We'll use a single consistent set of styles instead of overriding
        # the styles from style.css
        st.markdown("""
        <style>
        /* Global Light Theme Styles */
        body {
            background-color: var(--bg-darker);
            color: var(--text-primary);
        }
        
        .stApp {
            background: var(--bg-darker);
        }
        
        /* Sidebar Styling - From port 8502 */
        [data-testid="stSidebar"],
        .css-1d391kg,
        [data-testid="stSidebarNav"] {
            background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%) !important;
            padding: 1rem;
        }
        
        /* Sidebar title styling */
        [data-testid="stSidebar"] .st-emotion-cache-16idsys p,
        [data-testid="stSidebar"] .st-emotion-cache-16idsys span,
        [data-testid="stSidebar"] .st-emotion-cache-ue6h4q,
        [data-testid="stSidebar"] .st-emotion-cache-pkbazv,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        .css-1d391kg .st-emotion-cache-ue6h4q,
        [data-testid="stSidebarNav"] .st-emotion-cache-ue6h4q,
        .css-1d391kg .st-emotion-cache-eczf16,
        [data-testid="stSidebarNav"] .st-emotion-cache-eczf16,
        .css-1d391kg h1,
        [data-testid="stSidebarNav"] h1,
        .css-1d391kg p,
        [data-testid="stSidebarNav"] p,
        [data-testid="stSidebarNav"] .st-emotion-cache-18ni7ap {
            color: white !important;
            font-weight: 500;
        }
        
        /* Sidebar button styling - From port 8502 */
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button p,
        [data-testid="stSidebar"] button span,
        .css-1d391kg button,
        div[data-testid="stSidebarNav"] button,
        .css-1d391kg button p,
        div[data-testid="stSidebarNav"] button p,
        .css-1d391kg .st-emotion-cache-16idsys button,
        .css-1d391kg .st-emotion-cache-16idsys button p,
        .css-1d391kg .st-emotion-cache-16idsys button span,
        [data-testid="stSidebar"] .st-emotion-cache-16idsys button,
        [data-testid="stSidebar"] .st-emotion-cache-16idsys button p,
        [data-testid="stSidebar"] .st-emotion-cache-16idsys button span {
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 1rem !important;
            border-radius: 8px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            margin: 0.5rem 0 !important;
            width: 100% !important;
            text-align: left !important;
        }
        
        [data-testid="stSidebar"] button:hover,
        [data-testid="stSidebar"] button:active,
        [data-testid="stSidebar"] button:focus,
        .css-1d391kg button:hover,
        div[data-testid="stSidebarNav"] button:hover,
        .css-1d391kg .st-emotion-cache-16idsys button:hover,
        [data-testid="stSidebar"] .st-emotion-cache-16idsys button:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            transform: translateX(5px);
            color: white !important;
        }
        
        /* Card Styling */
        .stCard {
            background: var(--bg-dark);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            color: var(--text-primary);
        }
        
        .stCard:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        
        /* Button Styling - From port 8502 */
        .stButton > button,
        .st-emotion-cache-1vbkxwb p,
        .st-emotion-cache-19rxjzo button,
        .st-emotion-cache-7ym5gk,
        .st-emotion-cache-1erivf3 {
            background: var(--primary-gradient) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0, 105, 148, 0.2) !important;
            text-transform: none !important;
            letter-spacing: 0.5px !important;
            line-height: 1.4 !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #005577 0%, #0088B5 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 105, 148, 0.3) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 4px rgba(0, 105, 148, 0.2) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        

    def apply_input_styles(self):
        st.markdown("""
        <style>
        /* Input Fields */
        .stTextInput > div > div {
            background: white;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 8px;
            padding: 0;
        }
        
        .stTextInput > div > div:focus-within {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 1px var(--accent-color);
        }
        </style>
        """, unsafe_allow_html=True)
        
    def apply_text_input_styles(self):
        st.markdown("""
        <style>
        /* Streamlit Text Input Base Styles */
        .stTextInput > div > div {
            background: white;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 8px;
            padding: 0;
        }
        
        .stTextInput > div > div:focus-within {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 1px var(--accent-color);
        }
        
        /* Streamlit Password Input Styles */
        div[data-baseweb="base-input"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        div[data-baseweb="input"] {
            position: relative !important;
            background: white !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            margin: 8px 0 !important;
            padding: 0 !important;
        }
        
        div[data-baseweb="input"] input {
            background: transparent !important;
            border: none !important;
            color: var(--text-primary) !important;
            font-size: 14px !important;
            padding: 12px !important;
            height: 45px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            outline: none !important;
        }
        
        /* Remove all focus outlines */
        div[data-baseweb="input"]:focus,
        div[data-baseweb="input"] input:focus,
        div[data-baseweb="input"] div:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Single focus border on container */
        div[data-baseweb="input"]:focus-within {
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 1px var(--accent-color) !important;
        }
        
        /* Password Eye Icon */
        div[data-baseweb="input"] button {
            position: absolute !important;
            right: 12px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            background: none !important;
            border: none !important;
            padding: 4px !important;
            color: var(--text-secondary) !important;
            cursor: pointer !important;
            z-index: 2 !important;
            height: auto !important;
            min-height: auto !important;
            width: auto !important;
            min-width: auto !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Eye Icon Hover */
        div[data-baseweb="input"] button:hover,
        div[data-baseweb="input"] button:focus {
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            color: var(--accent-color) !important;
        }
        
        /* Remove any inner containers/wrappers styling */
        div[data-baseweb="input"] > div {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Admin Login Section */
        .stExpander {
            background: var(--bg-dark);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin: 10px 0;
            padding: 15px;
        }
        
        .stExpander .stButton > button {
            width: 100%;
            margin: 10px 0 5px 0;
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            height: 45px;
        }
        
        .stExpander .stButton > button:hover {
            background: var(--accent-color-dark);
        }
        
        /* File Uploader */
        .stFileUploader {
            background: white;
            border: 2px dashed var(--border-color);
            border-radius: 10px;
            padding: 2rem;
        }
        
        .stFileUploader:hover {
            border-color: var(--accent-color);
        }
        
        /* Metrics - Enhanced for consistency */
        .stMetric,
        [data-testid="stMetric"],
        .st-emotion-cache-10trblm,
        .st-emotion-cache-1wivap2 {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid var(--accent-color);
            color: var(--text-primary);
            transition: transform 0.3s ease;
        }
        
        .stMetric:hover,
        [data-testid="stMetric"]:hover,
        .st-emotion-cache-10trblm:hover,
        .st-emotion-cache-1wivap2:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0, 105, 148, 0.15);
        }
        
        /* Dashboard specific styling */
        .dashboard-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
            color: var(--text-primary);
            text-align: center;
        }
        
        .metric-card {
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--accent-color);
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 1rem;
            color: var(--text-secondary);
        }
        
        .chart-container {
            background-color: var(--bg-dark);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--text-primary);
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
        }
        
        h1 {
            color: var(--accent-color);
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }
        
        /* Progress Bars */
        .stProgress > div > div {
            background-color: var(--accent-color);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #f5f5f5;
            border: none;
            border-radius: 8px;
            color: var(--text-primary);
        }
        
        /* Plotly Charts */
        .js-plotly-plot {
            background: white !important;
        }
        
        /* Custom Components */
        .feature-card {
            background: var(--bg-light);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            border-color: var(--accent-color);
        }
        
        .resume-template {
            background: var(--bg-light);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem;
            border: 2px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .resume-template:hover {
            border-color: var(--accent-color);
            transform: scale(1.02);
        }
        
        /* Feedback Section */
        .feedback-card {
            background: var(--bg-light);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .feedback-card:hover {
            transform: translateX(5px);
            border-color: var(--accent-color);
        }
        
        /* About Section */
        .about-container {
            background: linear-gradient(135deg, var(--bg-light) 0%, var(--bg-dark) 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .about-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0,180,219,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }
        
        @keyframes rotate {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-darker);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-color);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.5s ease forwards;
        }
        
        /* Role card styling */
        .role-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #e1e4e8;
            transition: all 0.3s ease;
        }
        
        .role-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .role-card h3 {
            color: #2193b0;
            margin-bottom: 0.5rem;
        }
        
        /* Skills tag styling */
        .skill-tag {
            display: inline-block;
            background: rgba(33, 147, 176, 0.1);
            color: #2193b0;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            margin: 0.25rem;
            font-size: 0.875rem;
        }
        
        /* Template preview styling */
        .template-preview {
            border: 2px solid #e1e4e8;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, transparent 0%, rgba(255,255,255,0.1) 100%);
            z-index: 1;
        }

        .main-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 600;
            margin: 0;
            position: relative;
            z-index: 2;
        }

        /* Template Card Styles */
        .template-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 2rem;
            padding: 1rem;
        }

        .template-card {
            background: rgba(45, 45, 45, 0.9);
            border-radius: 20px;
            padding: 2rem;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .template-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            border-color: #4CAF50;
        }

        .template-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, transparent 0%, rgba(76,175,80,0.1) 100%);
            z-index: 1;
        }

        .template-icon {
            font-size: 3rem;
            color: #4CAF50;
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 2;
        }

        .template-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: white;
            margin-bottom: 1rem;
            position: relative;
            z-index: 2;
        }

        .template-description {
            color: #aaa;
            margin-bottom: 1.5rem;
            position: relative;
            z-index: 2;
            line-height: 1.6;
        }

        /* Feature List Styles */
        .feature-list {
            list-style: none;
            padding: 0;
            margin: 1.5rem 0;
            position: relative;
            z-index: 2;
        }

        .feature-item {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            color: #ddd;
            font-size: 0.95rem;
        }

        .feature-icon {
            color: #4CAF50;
            margin-right: 0.8rem;
            font-size: 1.1rem;
        }

        /* Button Styles */
        .action-button {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 1rem 2rem;
            border-radius: 50px;
            border: none;
            font-weight: 500;
            cursor: pointer;
            width: 100%;
            text-align: center;
            position: relative;
            overflow: hidden;
            z-index: 2;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .action-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(76,175,80,0.3);
        }

        .action-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
            transition: all 0.6s ease;
        }

        .action-button:hover::before {
            left: 100%;
        }

        /* Form Section Styles */
        .form-section {
            background: rgba(45, 45, 45, 0.9);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }

        .form-section-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: white;
            margin-bottom: 1.5rem;
            padding-bottom: 0.8rem;
            border-bottom: 2px solid #4CAF50;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            color: #ddd;
            font-weight: 500;
            margin-bottom: 0.8rem;
            display: block;
        }

        .form-input {
            width: 100%;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            background: rgba(30, 30, 30, 0.9);
            color: white;
            transition: all 0.3s ease;
        }

        .form-input:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 0 2px rgba(76,175,80,0.2);
            outline: none;
        }

        /* Skill Tags */
        .skill-tag-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .skill-tag {
            background: rgba(76,175,80,0.1);
            color: #4CAF50;
            padding: 0.6rem 1.2rem;
            border-radius: 50px;
            border: 1px solid #4CAF50;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .skill-tag:hover {
            background: #4CAF50;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76,175,80,0.2);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .template-container {
                grid-template-columns: 1fr;
            }

            .main-header {
                padding: 1.5rem;
            }

            .main-header h1 {
                font-size: 2rem;
            }

            .template-card {
                padding: 1.5rem;
            }

            .action-button {
                padding: 0.8rem 1.6rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)


    def export_to_excel(self):
        """Export resume data to Excel"""
        conn = get_database_connection()
        
        # Get resume data with analysis
        query = """
            SELECT 
                rd.name, rd.email, rd.phone, rd.linkedin, rd.github, rd.portfolio,
                rd.summary, rd.target_role, rd.target_category,
                rd.education, rd.experience, rd.projects, rd.skills,
                ra.ats_score, ra.keyword_match_score, ra.format_score, ra.section_score,
                ra.missing_skills, ra.recommendations,
                rd.created_at
            FROM resume_data rd
            LEFT JOIN resume_analysis ra ON rd.id = ra.resume_id
        """
        
        try:
            # Read data into DataFrame
            df = pd.read_sql_query(query, conn)
            
            # Create Excel writer object
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Resume Data')
            
            return output.getvalue()
        except Exception as e:
            print(f"Error exporting to Excel: {str(e)}")
            return None
        finally:
            conn.close()

    def render_dashboard(self):
        """Render the dashboard page with improved contrast for text and background"""
        st.markdown("<div style='background: #1E1E1E; color: #FFFFFF; padding: 20px; border-radius: 10px; margin: 20px 0;'>", unsafe_allow_html=True)
        self.dashboard_manager.render_dashboard()
        st.markdown("</div>", unsafe_allow_html=True)

    def render_empty_state(self, icon, message):
        """Render an empty state with icon and message"""
        return f"""
            <div style='text-align: center; padding: 2rem; color: #666;'>
                <i class='{icon}' style='font-size: 2rem; margin-bottom: 1rem; color: #00bfa5;'></i>
                <p style='margin: 0;'>{message}</p>
            </div>
        """

    def analyze_resume(self, resume_text):
        """Analyze resume and store results"""
        analytics = self.analyzer.analyze_resume(resume_text)
        st.session_state.analytics_data = analytics
        return analytics

    def handle_resume_upload(self):
        """Handle resume upload and analysis"""
        uploaded_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
        
        if uploaded_file is not None:
            try:
                # Extract text from resume
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = extract_text_from_docx(uploaded_file)
                
                # Store resume data
                st.session_state.resume_data = {
                    'filename': uploaded_file.name,
                    'content': resume_text,
                    'upload_time': datetime.now().isoformat()
                }
                
                # Analyze resume
                analytics = self.analyze_resume(resume_text)
                
                return True
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")
                return False
        return False

    def render_builder(self):
        st.title("Resume Builder 📝")
        st.write("Create your professional resume")
        
        # Template selection
        template_options = ["Modern", "Professional", "Minimal", "Creative"]
        selected_template = st.selectbox("Select Resume Template", template_options)
        st.success(f"🎨 Currently using: {selected_template} Template")

        # Personal Information
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            # Get existing values from session state
            existing_name = st.session_state.form_data['personal_info']['full_name']
            existing_email = st.session_state.form_data['personal_info']['email']
            existing_phone = st.session_state.form_data['personal_info']['phone']
            
            # Input fields with existing values
            full_name = st.text_input("Full Name", value=existing_name)
            email = st.text_input("Email", value=existing_email, key="email_input")
            phone = st.text_input("Phone", value=existing_phone)

            # Immediately update session state after email input
            if 'email_input' in st.session_state:
                st.session_state.form_data['personal_info']['email'] = st.session_state.email_input
        
        with col2:
            # Get existing values from session state
            existing_location = st.session_state.form_data['personal_info']['location']
            existing_linkedin = st.session_state.form_data['personal_info']['linkedin']
            existing_portfolio = st.session_state.form_data['personal_info']['portfolio']
            
            # Input fields with existing values
            location = st.text_input("Location", value=existing_location)
            linkedin = st.text_input("LinkedIn URL", value=existing_linkedin)
            portfolio = st.text_input("Portfolio Website", value=existing_portfolio)

        # Update personal info in session state
        st.session_state.form_data['personal_info'] = {
            'full_name': full_name,
            'email': email,
            'phone': phone,
            'location': location,
            'linkedin': linkedin,
            'portfolio': portfolio
        }

        # Professional Summary
        st.subheader("Professional Summary")
        summary = st.text_area("Professional Summary", value=st.session_state.form_data.get('summary', ''), height=150,
                             help="Write a brief summary highlighting your key skills and experience")
        
        # Experience Section
        st.subheader("Work Experience")
        if 'experiences' not in st.session_state.form_data:
            st.session_state.form_data['experiences'] = []
            
        if st.button("Add Experience"):
            st.session_state.form_data['experiences'].append({
                'company': '',
                'position': '',
                'start_date': '',
                'end_date': '',
                'description': '',
                'responsibilities': [],
                'achievements': []
            })
        
        for idx, exp in enumerate(st.session_state.form_data['experiences']):
            with st.expander(f"Experience {idx + 1}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    exp['company'] = st.text_input("Company Name", key=f"company_{idx}", value=exp.get('company', ''))
                    exp['position'] = st.text_input("Position", key=f"position_{idx}", value=exp.get('position', ''))
                with col2:
                    exp['start_date'] = st.text_input("Start Date", key=f"start_date_{idx}", value=exp.get('start_date', ''))
                    exp['end_date'] = st.text_input("End Date", key=f"end_date_{idx}", value=exp.get('end_date', ''))
                
                exp['description'] = st.text_area("Role Overview", key=f"desc_{idx}", 
                                                value=exp.get('description', ''),
                                                help="Brief overview of your role and impact")
                
                # Responsibilities
                st.markdown("##### Key Responsibilities")
                resp_text = st.text_area("Enter responsibilities (one per line)", 
                                       key=f"resp_{idx}",
                                       value='\n'.join(exp.get('responsibilities', [])),
                                       height=100,
                                       help="List your main responsibilities, one per line")
                exp['responsibilities'] = [r.strip() for r in resp_text.split('\n') if r.strip()]
                
                # Achievements
                st.markdown("##### Key Achievements")
                achv_text = st.text_area("Enter achievements (one per line)", 
                                       key=f"achv_{idx}",
                                       value='\n'.join(exp.get('achievements', [])),
                                       height=100,
                                       help="List your notable achievements, one per line")
                exp['achievements'] = [a.strip() for a in achv_text.split('\n') if a.strip()]
                
                if st.button("Remove Experience", key=f"remove_exp_{idx}"):
                    st.session_state.form_data['experiences'].pop(idx)
                    st.rerun()
        
        # Projects Section
        st.subheader("Projects")
        if 'projects' not in st.session_state.form_data:
            st.session_state.form_data['projects'] = []
            
        if st.button("Add Project"):
            st.session_state.form_data['projects'].append({
                'name': '',
                'technologies': '',
                'description': '',
                'responsibilities': [],
                'achievements': [],
                'link': ''
            })
        
        for idx, proj in enumerate(st.session_state.form_data['projects']):
            with st.expander(f"Project {idx + 1}", expanded=True):
                proj['name'] = st.text_input("Project Name", key=f"proj_name_{idx}", value=proj.get('name', ''))
                proj['technologies'] = st.text_input("Technologies Used", key=f"proj_tech_{idx}", 
                                                   value=proj.get('technologies', ''),
                                                   help="List the main technologies, frameworks, and tools used")
                
                proj['description'] = st.text_area("Project Overview", key=f"proj_desc_{idx}", 
                                                 value=proj.get('description', ''),
                                                 help="Brief overview of the project and its goals")
                
                # Project Responsibilities
                st.markdown("##### Key Responsibilities")
                proj_resp_text = st.text_area("Enter responsibilities (one per line)", 
                                            key=f"proj_resp_{idx}",
                                            value='\n'.join(proj.get('responsibilities', [])),
                                            height=100,
                                            help="List your main responsibilities in the project")
                proj['responsibilities'] = [r.strip() for r in proj_resp_text.split('\n') if r.strip()]
                
                # Project Achievements
                st.markdown("##### Key Achievements")
                proj_achv_text = st.text_area("Enter achievements (one per line)", 
                                            key=f"proj_achv_{idx}",
                                            value='\n'.join(proj.get('achievements', [])),
                                            height=100,
                                            help="List the project's key achievements and your contributions")
                proj['achievements'] = [a.strip() for a in proj_achv_text.split('\n') if a.strip()]
                
                proj['link'] = st.text_input("Project Link (optional)", key=f"proj_link_{idx}", 
                                           value=proj.get('link', ''),
                                           help="Link to the project repository, demo, or documentation")
                
                if st.button("Remove Project", key=f"remove_proj_{idx}"):
                    st.session_state.form_data['projects'].pop(idx)
                    st.rerun()
        
        # Education Section
        st.subheader("Education")
        if 'education' not in st.session_state.form_data:
            st.session_state.form_data['education'] = []
            
        if st.button("Add Education"):
            st.session_state.form_data['education'].append({
                'school': '',
                'degree': '',
                'field': '',
                'graduation_date': '',
                'gpa': '',
                'achievements': []
            })
        
        for idx, edu in enumerate(st.session_state.form_data['education']):
            with st.expander(f"Education {idx + 1}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    edu['school'] = st.text_input("School/University", key=f"school_{idx}", value=edu.get('school', ''))
                    edu['degree'] = st.text_input("Degree", key=f"degree_{idx}", value=edu.get('degree', ''))
                with col2:
                    edu['field'] = st.text_input("Field of Study", key=f"field_{idx}", value=edu.get('field', ''))
                    edu['graduation_date'] = st.text_input("Graduation Date", key=f"grad_date_{idx}", 
                                                         value=edu.get('graduation_date', ''))
                
                edu['gpa'] = st.text_input("GPA (optional)", key=f"gpa_{idx}", value=edu.get('gpa', ''))
                
                # Educational Achievements
                st.markdown("##### Achievements & Activities")
                edu_achv_text = st.text_area("Enter achievements (one per line)", 
                                           key=f"edu_achv_{idx}",
                                           value='\n'.join(edu.get('achievements', [])),
                                           height=100,
                                           help="List academic achievements, relevant coursework, or activities")
                edu['achievements'] = [a.strip() for a in edu_achv_text.split('\n') if a.strip()]
                
                if st.button("Remove Education", key=f"remove_edu_{idx}"):
                    st.session_state.form_data['education'].pop(idx)
                    st.rerun()
        
        # Skills Section
        st.subheader("Skills")
        if 'skills_categories' not in st.session_state.form_data:
            st.session_state.form_data['skills_categories'] = {
                'technical': [],
                'soft': [],
                'languages': [],
                'tools': []
            }
        
        col1, col2 = st.columns(2)
        with col1:
            tech_skills = st.text_area("Technical Skills (one per line)", 
                                     value='\n'.join(st.session_state.form_data['skills_categories']['technical']),
                                     height=150,
                                     help="Programming languages, frameworks, databases, etc.")
            st.session_state.form_data['skills_categories']['technical'] = [s.strip() for s in tech_skills.split('\n') if s.strip()]
            
            soft_skills = st.text_area("Soft Skills (one per line)", 
                                     value='\n'.join(st.session_state.form_data['skills_categories']['soft']),
                                     height=150,
                                     help="Leadership, communication, problem-solving, etc.")
            st.session_state.form_data['skills_categories']['soft'] = [s.strip() for s in soft_skills.split('\n') if s.strip()]
        
        with col2:
            languages = st.text_area("Languages (one per line)", 
                                   value='\n'.join(st.session_state.form_data['skills_categories']['languages']),
                                   height=150,
                                   help="Programming or human languages with proficiency level")
            st.session_state.form_data['skills_categories']['languages'] = [l.strip() for l in languages.split('\n') if l.strip()]
            
            tools = st.text_area("Tools & Technologies (one per line)", 
                               value='\n'.join(st.session_state.form_data['skills_categories']['tools']),
                               height=150,
                               help="Development tools, software, platforms, etc.")
            st.session_state.form_data['skills_categories']['tools'] = [t.strip() for t in tools.split('\n') if t.strip()]
        
        # Update form data in session state
        st.session_state.form_data.update({
            'summary': summary
        })
        
        # Generate Resume button
        if st.button("Generate Resume 📄", type="primary"):
            print("Validating form data...")
            print(f"Session state form data: {st.session_state.form_data}")
            print(f"Email input value: {st.session_state.get('email_input', '')}")
            
            # Get the current values from form
            current_name = st.session_state.form_data['personal_info']['full_name'].strip()
            current_email = st.session_state.email_input if 'email_input' in st.session_state else ''
            
            print(f"Current name: {current_name}")
            print(f"Current email: {current_email}")
            
            # Validate required fields
            if not current_name:
                st.error("⚠️ Please enter your full name.")
                return
            
            if not current_email:
                st.error("⚠️ Please enter your email address.")
                return
                
            # Update email in form data one final time
            st.session_state.form_data['personal_info']['email'] = current_email
            
            try:
                print("Preparing resume data...")
                # Prepare resume data with current form values
                resume_data = {
                    "personal_info": st.session_state.form_data['personal_info'],
                    "summary": st.session_state.form_data.get('summary', '').strip(),
                    "experience": st.session_state.form_data.get('experiences', []),
                    "education": st.session_state.form_data.get('education', []),
                    "projects": st.session_state.form_data.get('projects', []),
                    "skills": st.session_state.form_data.get('skills_categories', {
                        'technical': [],
                        'soft': [],
                        'languages': [],
                        'tools': []
                    }),
                    "template": selected_template
                }
                
                print(f"Resume data prepared: {resume_data}")
                
                try:
                    # Generate resume
                    resume_buffer = self.builder.generate_resume(resume_data)
                    if resume_buffer:
                        try:
                            # Save resume data to database
                            save_resume_data(resume_data)
                            
                            # Offer the resume for download
                            st.success("✅ Resume generated successfully!")
                            st.download_button(
                                label="Download Resume 📥",
                                data=resume_buffer,
                                file_name=f"{current_name.replace(' ', '_')}_resume.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        except Exception as db_error:
                            print(f"Warning: Failed to save to database: {str(db_error)}")
                            # Still allow download even if database save fails
                            st.warning("⚠️ Resume generated but couldn't be saved to database")
                            st.download_button(
                                label="Download Resume 📥",
                                data=resume_buffer,
                                file_name=f"{current_name.replace(' ', '_')}_resume.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                    else:
                        st.error("❌ Failed to generate resume. Please try again.")
                        print("Resume buffer was None")
                except Exception as gen_error:
                    print(f"Error during resume generation: {str(gen_error)}")
                    print(f"Full traceback: {traceback.format_exc()}")
                    st.error(f"❌ Error generating resume: {str(gen_error)}")
                        
            except Exception as e:
                print(f"Error preparing resume data: {str(e)}")
                print(f"Full traceback: {traceback.format_exc()}")
                st.error(f"❌ Error preparing resume data: {str(e)}")
    
    def render_about(self):
        """Render the about page with updated light theme"""
        from ui_components import apply_modern_styles
        import base64, os
        
        def get_image_as_base64(file_path):
            try:
                with open(file_path, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode()
                    return f"data:image/jpeg;base64,{encoded}"
            except:
                return None
        
        image_path = os.path.join(os.path.dirname(__file__), "assets", "124852522.jpeg")
        image_base64 = get_image_as_base64(image_path)
        
        apply_modern_styles()
        
        # Updated CSS: light backgrounds (#f7f7f7) with dark text (#333333)
        st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                .profile-section, .vision-section, .feature-card {
                    text-align: center;
                    padding: 2rem;
                    background: #f7f7f7;
                    border-radius: 10px;
                    margin: 2rem auto;
                    max-width: 800px;
                    color: #333333;
                }
                
                .profile-image {
                    width: 200px;
                    height: 200px;
                    border-radius: 50%;
                    margin: 0 auto 1.5rem;
                    display: block;
                    object-fit: cover;
                    border: 4px solid #0077b5;
                }
                
                .profile-name {
                    font-size: 2.5rem;
                    color: #333333;
                    margin-bottom: 0.5rem;
                }
                
                .profile-title {
                    font-size: 1.2rem;
                    color: #0077b5;
                    margin-bottom: 1.5rem;
                }
                
                .social-links {
                    display: flex;
                    justify-content: center;
                    gap: 1.5rem;
                    margin: 2rem 0;
                }
                
                .social-link {
                    font-size: 2rem;
                    color: #0077b5;
                    transition: all 0.3s ease;
                    padding: 0.5rem;
                    border-radius: 50%;
                    background: #e6f2fa;
                    width: 60px;
                    height: 60px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-decoration: none;
                }
                
                .social-link:hover {
                    transform: translateY(-5px);
                    background: #0077b5;
                    color: white;
                    box-shadow: 0 5px 15px rgba(0, 119, 181, 0.3);
                }
                
                .bio-text {
                    color: #333333;
                    line-height: 1.8;
                    font-size: 1.1rem;
                    margin-top: 2rem;
                    text-align: left;
                }

                .vision-text {
                    color: #333333;
                    line-height: 1.8;
                    font-size: 1.1rem;
                    font-style: italic;
                    margin: 1.5rem 0;
                    text-align: left;
                }

                .vision-icon {
                    font-size: 2.5rem;
                    color: #0077b5;
                    margin-bottom: 1rem;
                }

                .vision-title {
                    font-size: 2rem;
                    color: #333333;
                    margin-bottom: 1rem;
                }

                .features-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 2rem;
                    margin: 2rem auto;
                    max-width: 1200px;
                }

                .feature-card {
                    padding: 2rem;
                    margin: 0;
                    background: #f7f7f7;
                    color: #333333;
                }

                .feature-icon {
                    font-size: 2.5rem;
                    color: #0077b5;
                    margin-bottom: 1rem;
                }

                .feature-title {
                    font-size: 1.5rem;
                    color: #333333;
                    margin: 1rem 0;
                }

                .feature-description {
                    color: #333333;
                    line-height: 1.6;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="hero-section">
                <h1 class="hero-title">About Smart Resume AI</h1>
                <p class="hero-subtitle">A powerful AI-driven platform for optimizing your resume</p>
            </div>
        """, unsafe_allow_html=True)
        # Profile Section
        
        # Vision Section
        st.markdown("""
            <div class="vision-section">
                <i class="fas fa-lightbulb vision-icon"></i>
                <h2 class="vision-title">Our Vision</h2>
                <p class="vision-text">
                    "Smart Resume AI represents my vision of democratizing career advancement through technology. 
                    By combining cutting-edge AI with intuitive design, this platform empowers job seekers at 
                    every career stage to showcase their true potential and stand out in today's competitive job market."
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Features Section
        st.markdown("""
            <div class="features-grid">
                <div class="feature-card">
                    <i class="fas fa-robot feature-icon"></i>
                    <h3 class="feature-title">AI-Powered Analysis</h3>
                    <p class="feature-description">
                        Advanced AI algorithms provide detailed insights and suggestions to optimize your resume for maximum impact.
                    </p>
                </div>
                <div class="feature-card">
                    <i class="fas fa-chart-line feature-icon"></i>
                    <h3 class="feature-title">Data-Driven Insights</h3>
                    <p class="feature-description">
                        Make informed decisions with our analytics-based recommendations and industry insights.
                    </p>
                </div>
                <div class="feature-card">
                    <i class="fas fa-shield-alt feature-icon"></i>
                    <h3 class="feature-title">Privacy First</h3>
                    <p class="feature-description">
                        Your data security is our priority. We ensure your information is always protected and private.
                    </p>
                </div>
            </div>
            <div style="text-align: center; margin: 3rem 0;">
                <a href="?page=analyzer" class="cta-button">
                    Start Your Journey
                    <i class="fas fa-arrow-right" style="margin-left: 10px;"></i>
                </a>
            </div>
        """, unsafe_allow_html=True)
    
    def render_analyzer(self):
        """Render the resume analyzer page"""
        apply_modern_styles()
        
        # Page Header
        page_header(
            "Resume Analyzer",
            "Get instant AI-powered feedback to optimize your resume"
        )
        
        # Job Role Selection (Updated to light background and dark text)
        categories = list(self.job_roles.keys())
        selected_category = st.selectbox("Job Category", categories)
        
        roles = list(self.job_roles[selected_category].keys())
        selected_role = st.selectbox("Specific Role", roles)
        
        role_info = self.job_roles[selected_category][selected_role]
        
        st.markdown(f"""
        <div style='background-color: #ffffff; color: #333333; padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <h3>{selected_role}</h3>
            <p>{role_info['description']}</p>
            <h4>Required Skills:</h4>
            <p>{', '.join(role_info['required_skills'])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File Upload
        uploaded_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
        
        st.markdown(
            self.render_empty_state(
            "fas fa-cloud-upload-alt",
            "Upload your resume to get started with AI-powered analysis"
            ),
            unsafe_allow_html=True
        )
        if uploaded_file:
            with st.spinner("Analyzing your document..."):
                # Get file content
                text = ""
                try:
                    if uploaded_file.type == "application/pdf":
                        text = self.analyzer.extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = self.analyzer.extract_text_from_docx(uploaded_file)
                    else:
                        text = uploaded_file.getvalue().decode()
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return

                
                # Analyze the document
                analysis = self.analyzer.analyze_resume({'raw_text': text}, role_info)
                
                # Save resume data to database
                resume_data = {
                    'personal_info': {
                        'name': analysis.get('name', ''),
                        'email': analysis.get('email', ''),
                        'phone': analysis.get('phone', ''),
                        'linkedin': analysis.get('linkedin', ''),
                        'github': analysis.get('github', ''),
                        'portfolio': analysis.get('portfolio', '')
                    },
                    'summary': analysis.get('summary', ''),
                    'target_role': selected_role,
                    'target_category': selected_category,
                    'education': analysis.get('education', []),
                    'experience': analysis.get('experience', []),
                    'projects': analysis.get('projects', []),
                    'skills': analysis.get('skills', []),
                    'template': ''
                }
                
                # Save to database
                try:
                    resume_id = save_resume_data(resume_data)
                    
                    # Save analysis data
                    analysis_data = {
                        'resume_id': resume_id,
                        'ats_score': analysis['ats_score'],
                        'keyword_match_score': analysis['keyword_match']['score'],
                        'format_score': analysis['format_score'],
                        'section_score': analysis['section_score'],
                        'missing_skills': ','.join(analysis['keyword_match']['missing_skills']),
                        'recommendations': ','.join(analysis['suggestions'])
                    }
                    save_analysis_data(resume_id, analysis_data)
                    st.success("Resume data saved successfully!")
                except Exception as e:
                    st.error(f"Error saving to database: {str(e)}")
                    print(f"Database error: {e}")
                
                # Show results based on document type
                if analysis.get('document_type') != 'resume':
                    st.error(f"⚠️ This appears to be a {analysis['document_type']} document, not a resume!")
                    st.warning("Please upload a proper resume for ATS analysis.")
                    return                
                # Display results in a modern card layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # ATS Score Card with circular progress
                    st.markdown("""
                    <div class="feature-card">
                        <h2>ATS Score</h2>
                        <div style="position: relative; width: 150px; height: 150px; margin: 0 auto;">
                            <div style="
                                position: absolute;
                                width: 150px;
                                height: 150px;
                                border-radius: 50%;
                                background: conic-gradient(
                                    #4CAF50 0% {score}%,
                                    #2c2c2c {score}% 100%
                                );
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            ">
                                <div style="
                                    width: 120px;
                                    height: 120px;
                                    background: #1a1a1a;
                                    border-radius: 50%;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    font-size: 24px;
                                    font-weight: bold;
                                    color: {color};
                                ">
                                    {score}
                                </div>
                            </div>
                        </div>
                        <div style="text-align: center; margin-top: 10px;">
                            <span style="
                                font-size: 1.2em;
                                color: {color};
                                font-weight: bold;
                            ">
                                {status}
                            </span>
                        </div>
                    """.format(
                        score=analysis['ats_score'],
                        color='#4CAF50' if analysis['ats_score'] >= 80 else '#FFA500' if analysis['ats_score'] >= 60 else '#FF4444',
                        status='Excellent' if analysis['ats_score'] >= 80 else 'Good' if analysis['ats_score'] >= 60 else 'Needs Improvement'
                    ), unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                                        
                    # self.display_analysis_results(analysis_results)

                    # Skills Match Card
                    st.markdown("""
                    <div class="feature-card">
                        <h2>Skills Match</h2>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Keyword Match", f"{int(analysis.get('keyword_match', {}).get('score', 0))}%")
                    
                    if analysis['keyword_match']['missing_skills']:
                        st.markdown("#### Missing Skills:")
                        for skill in analysis['keyword_match']['missing_skills']:
                            st.markdown(f"- {skill}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    # Format Score Card
                    st.markdown("""
                    <div class="feature-card">
                        <h2>Format Analysis</h2>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Format Score", f"{int(analysis.get('format_score', 0))}%")
                    st.metric("Section Score", f"{int(analysis.get('section_score', 0))}%")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Suggestions Card with improved UI
                    st.markdown("""
                    <div class="feature-card">
                        <h2>📋 Resume Improvement Suggestions</h2>
                    """, unsafe_allow_html=True)
                    
                    # Contact Section
                    if analysis.get('contact_suggestions'):
                        st.markdown("""
                        <div style='background-color: #ffffff; color: #333333; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h3 style='color: #4CAF50; margin-bottom: 10px;'>📞 Contact Information</h3>
                            <ul style='list-style-type: none; padding-left: 0;'>
                        """, unsafe_allow_html=True)
                        for suggestion in analysis.get('contact_suggestions', []):
                            st.markdown(f"<li style='margin-bottom: 8px;'>✓ {suggestion}</li>", unsafe_allow_html=True)
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Summary Section
                    if analysis.get('summary_suggestions'):
                        st.markdown("""
                        <div style='background-color: #ffffff; color: #333333; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h3 style='color: #4CAF50; margin-bottom: 10px;'>📝 Professional Summary</h3>
                            <ul style='list-style-type: none; padding-left: 0;'>
                        """, unsafe_allow_html=True)
                        for suggestion in analysis.get('summary_suggestions', []):
                            st.markdown(f"<li style='margin-bottom: 8px;'>✓ {suggestion}</li>", unsafe_allow_html=True)
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Skills Section
                    if analysis.get('skills_suggestions') or analysis['keyword_match']['missing_skills']:
                        st.markdown("""
                        <div style='background-color: #ffffff; color: #333333; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h3 style='color: #4CAF50; margin-bottom: 10px;'>🎯 Skills</h3>
                            <ul style='list-style-type: none; padding-left: 0;'>
                        """, unsafe_allow_html=True)
                        for suggestion in analysis.get('skills_suggestions', []):
                            st.markdown(f"<li style='margin-bottom: 8px;'>✓ {suggestion}</li>", unsafe_allow_html=True)
                        if analysis['keyword_match']['missing_skills']:
                            st.markdown("<li style='margin-bottom: 8px;'>✓ Consider adding these relevant skills:</li>", unsafe_allow_html=True)
                            for skill in analysis['keyword_match']['missing_skills']:
                                st.markdown(f"<li style='margin-left: 20px; margin-bottom: 4px;'>• {skill}</li>", unsafe_allow_html=True)
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Experience Section
                    if analysis.get('experience_suggestions'):
                        st.markdown("""
                        <div style='background-color: #ffffff; color: #333333; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h3 style='color: #4CAF50; margin-bottom: 10px;'>💼 Work Experience</h3>
                            <ul style='list-style-type: none; padding-left: 0;'>
                        """, unsafe_allow_html=True)
                        for suggestion in analysis.get('experience_suggestions', []):
                            st.markdown(f"<li style='margin-bottom: 8px;'>✓ {suggestion}</li>", unsafe_allow_html=True)
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Education Section
                    if analysis.get('education_suggestions'):
                        st.markdown("""
                        <div style='background-color: #ffffff; color: #333333; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h3 style='color: #4CAF50; margin-bottom: 10px;'>🎓 Education</h3>
                            <ul style='list-style-type: none; padding-left: 0;'>
                        """, unsafe_allow_html=True)
                        for suggestion in analysis.get('education_suggestions', []):
                            st.markdown(f"<li style='margin-bottom: 8px;'>✓ {suggestion}</li>", unsafe_allow_html=True)
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # General Formatting Suggestions
                    if analysis.get('format_suggestions'):
                        st.markdown("""
                        <div style='background-color: #ffffff; color: #333333; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h3 style='color: #4CAF50; margin-bottom: 10px;'>📄 Formatting</h3>
                            <ul style='list-style-type: none; padding-left: 0;'>
                        """, unsafe_allow_html=True)
                        for suggestion in analysis.get('format_suggestions', []):
                            st.markdown(f"<li style='margin-bottom: 8px;'>✓ {suggestion}</li>", unsafe_allow_html=True)
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                

                
                # Course Recommendations
                st.markdown("""
                <div class="feature-card">
                    <h2>📚 Recommended Courses</h2>
                """, unsafe_allow_html=True)
                
                # Get courses based on role and category
                courses = get_courses_for_role(selected_role)
                if not courses:
                    category = get_category_for_role(selected_role)
                    courses = COURSES_BY_CATEGORY.get(category, {}).get(selected_role, [])
                
                # Display courses in a grid
                cols = st.columns(2)
                for i, course in enumerate(courses[:6]):  # Show top 6 courses
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div style='background-color: #ffffff; color: #333333; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h4>{course[0]}</h4>
                            <a href='{course[1]}' target='_blank'>View Course</a>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Learning Resources
                st.markdown("""
                <div class="feature-card">
                    <h2>📺 Helpful Videos</h2>
                """, unsafe_allow_html=True)
                
                tab1, tab2 = st.tabs(["Resume Tips", "Interview Tips"])
                
                with tab1:
                    # Resume Videos
                    for category, videos in RESUME_VIDEOS.items():
                        st.subheader(category)
                        cols = st.columns(2)
                        for i, video in enumerate(videos):
                            with cols[i % 2]:
                                st.video(video[1])
                
                with tab2:
                    # Interview Videos
                    for category, videos in INTERVIEW_VIDEOS.items():
                        st.subheader(category)
                        cols = st.columns(2)
                        for i, video in enumerate(videos):
                            with cols[i % 2]:
                                st.video(video[1])
                
                st.markdown("</div>", unsafe_allow_html=True)
                
        # Close the page container
        st.markdown('</div>', unsafe_allow_html=True)

    def render_job_search(self):
        """Render the job search page"""
        render_job_search()

    def render_feedback_page(self):
        """Render the feedback page"""
        st.markdown("""
            <style>
            .feedback-header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(90deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1));
                border-radius: 10px;
                margin-bottom: 30px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="feedback-header">
                <h1>📣 Your Voice Matters!</h1>
                <p>Help us improve Smart Resume AI with your valuable feedback</p>
            </div>
        """, unsafe_allow_html=True)

        # Initialize feedback manager
        feedback_manager = FeedbackManager()
        
        # Create tabs for form and statistics
        form_tab, stats_tab = st.tabs(["Share Feedback", "Feedback Overview"])
        
        with form_tab:
            feedback_manager.render_feedback_form()
            
        with stats_tab:
            feedback_manager.render_feedback_stats()

    def render_home(self):
        apply_modern_styles()
        
        # Hero Section
        hero_section(
            "Smart Resume AI",
            "Transform your career with AI-powered resume analysis and building. Get personalized insights and create professional resumes that stand out."
        )
        
        # Features Section
        st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
        
        feature_card(
            "fas fa-robot",
            "AI-Powered Analysis",
            "Get instant feedback on your resume with advanced AI analysis that identifies strengths and areas for improvement."
        )
        
        feature_card(
            "fas fa-magic",
            "Smart Resume Builder",
            "Create professional resumes with our intelligent builder that suggests optimal content and formatting."
        )
        
        feature_card(
            "fas fa-chart-line",
            "Career Insights",
            "Access detailed analytics and personalized recommendations to enhance your career prospects."
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Call-to-Action with Streamlit navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Get Started", key="get_started_btn", 
                        help="Click to start analyzing your resume",
                        type="primary",
                        use_container_width=True):
                cleaned_name = "🔍 RESUME ANALYZER".lower().replace(" ", "_").replace("🔍", "").strip()
                st.session_state.page = cleaned_name
                st.rerun()

    def main(self):
        """Main application entry point"""
        self.apply_global_styles()
        
        # Admin login/logout in sidebar
        with st.sidebar:
            st_lottie(self.load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_xyadoh9h.json"), height=200, key="sidebar_animation")
            st.title("Smart Resume AI")
            st.markdown("---")
            
            # Navigation buttons
            for page_name in self.pages.keys():
                if st.button(page_name, use_container_width=True):
                    cleaned_name = page_name.lower().replace(" ", "_").replace("🏠", "").replace("🔍", "").replace("📝", "").replace("📊", "").replace("🎯", "").replace("💬", "").replace("ℹ️", "").strip()
                    st.session_state.page = cleaned_name
                    st.rerun()

            # Add some space before admin login
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Admin Login/Logout section at bottom
            if st.session_state.get('is_admin', False):
                st.success(f"Logged in as: {st.session_state.get('current_admin_email')}")
                if st.button("Logout", key="logout_button"):
                    try:
                        log_admin_action(st.session_state.get('current_admin_email'), "logout")
                        st.session_state.is_admin = False
                        st.session_state.current_admin_email = None
                        st.success("Logged out successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during logout: {str(e)}")
            else:
                with st.expander("👤 Admin Login"):
                    admin_email_input = st.text_input("Email", key="admin_email_input")
                    admin_password = st.text_input("Password", type="password", key="admin_password_input")
                    if st.button("Login", key="login_button"):
                            try:
                                if verify_admin(admin_email_input, admin_password):
                                    st.session_state.is_admin = True
                                    st.session_state.current_admin_email = admin_email_input
                                    log_admin_action(admin_email_input, "login")
                                    st.success("Logged in successfully!")
                                    st.rerun()
                                else:
                                    st.error("Invalid credentials")
                            except Exception as e:
                                st.error(f"Error during login: {str(e)}")
        
        # Force home page on first load
        if 'initial_load' not in st.session_state:
            st.session_state.initial_load = True
            st.session_state.page = 'home'
            st.rerun()
        
        # Get current page and render it
        current_page = st.session_state.get('page', 'home')
        
        # Create a mapping of cleaned page names to original names
        page_mapping = {name.lower().replace(" ", "_").replace("🏠", "").replace("🔍", "").replace("📝", "").replace("📊", "").replace("🎯", "").replace("💬", "").replace("ℹ️", "").strip(): name 
                       for name in self.pages.keys()}
        
        # Render the appropriate page
        if current_page in page_mapping:
            self.pages[page_mapping[current_page]]()
        else:
            # Default to home page if invalid page
            self.render_home()
    
if __name__ == "__main__":
    app = ResumeApp()
    app.main()
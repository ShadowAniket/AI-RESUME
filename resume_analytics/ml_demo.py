import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from datetime import datetime

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our ML modules
from resume_analytics.ml_models import ResumeMLModel, download_resume_dataset
from resume_analytics.ml_integration import MLResumeAnalyzer

# Page configuration
st.set_page_config(
    page_title="Resume ML Model Demo",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2196F3;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 10px;
        flex: 1;
        min-width: 200px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #666;
    }
    .suggestion-item {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">Resume ML Model Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("ML Model Options")
    
    # Model training section
    st.sidebar.markdown("### Model Training")
    if st.sidebar.button("Train Models with Synthetic Data"):
        with st.spinner("Training models with synthetic data..."):
            # Create progress bar
            progress_bar = st.sidebar.progress(0)
            
            # Download dataset
            progress_bar.progress(10)
            st.sidebar.info("Generating synthetic dataset...")
            dataset_path = download_resume_dataset('resume_dataset.csv')
            
            # Create model
            progress_bar.progress(30)
            st.sidebar.info("Initializing ML model...")
            model = ResumeMLModel()
            
            # Train model
            progress_bar.progress(50)
            st.sidebar.info("Training models...")
            model.train_models(dataset_path)
            
            # Complete
            progress_bar.progress(100)
            st.sidebar.success("Models trained successfully!")
    
    # Main content
    tabs = st.tabs(["Resume Analysis", "Model Performance", "Dataset Exploration"])
    
    # Resume Analysis Tab
    with tabs[0]:
        st.markdown('<h2 class="section-header">Resume Analysis with ML</h2>', unsafe_allow_html=True)
        
        # Job description input
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Job Description")
        job_description = st.text_area(
            "Enter the job description",
            """Senior Software Engineer

We are looking for a Senior Software Engineer with strong experience in Python, 
JavaScript, and cloud technologies. The ideal candidate will have experience with 
React, Node.js, and database optimization. Machine learning knowledge is a plus.

Requirements:
- 5+ years of software development experience
- Strong knowledge of Python and JavaScript
- Experience with React and Node.js
- Database optimization skills
- CI/CD pipeline implementation
- Cloud platform experience (AWS, Azure, or GCP)
""",
            height=200
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Resume input
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Resume Text")
        resume_quality = st.select_slider(
            "Resume Quality",
            options=["Poor", "Average", "Good", "Excellent"],
            value="Good"
        )
        
        # Generate sample resume based on quality
        if resume_quality == "Poor":
            sample_resume = """
John Doe
john@example.com

I am looking for a job in software development.

Worked at Company A for 2 years
Graduated from University B

Skills: Python
"""
        elif resume_quality == "Average":
            sample_resume = """
JOHN DOE
Email: john@example.com
Phone: (123) 456-7890

SUMMARY
Software developer with 3 years of experience in web development.

EXPERIENCE
Software Developer
Company A | 2018 - 2021
- Developed web applications using JavaScript
- Worked on database optimization

EDUCATION
Bachelor of Computer Science
University B | 2014 - 2018

SKILLS
Python, JavaScript, HTML, CSS, SQL
"""
        elif resume_quality == "Good":
            sample_resume = """
JOHN DOE
Email: john@example.com | Phone: (123) 456-7890 | LinkedIn: linkedin.com/in/johndoe

SUMMARY
Senior Software Engineer with 5 years of experience in full-stack development and cloud technologies.
Proficient in Python, JavaScript, React, and Node.js with a focus on building scalable applications.

EXPERIENCE
SENIOR SOFTWARE ENGINEER
Tech Solutions Inc. | 2019 - Present
• Developed and maintained RESTful APIs using Node.js and Express
• Implemented CI/CD pipelines reducing deployment time by 50%
• Optimized database queries resulting in 30% performance improvement
• Collaborated with cross-functional teams to deliver features on schedule

SOFTWARE DEVELOPER
Innovate Systems | 2017 - 2019
• Built responsive web applications using React and Redux
• Implemented automated testing achieving 80% code coverage
• Participated in code reviews and mentored junior developers

EDUCATION
Master of Computer Science
Tech University | 2015 - 2017

Bachelor of Science in Computer Engineering
State University | 2011 - 2015

SKILLS
Languages: Python, JavaScript, HTML/CSS, SQL
Frameworks: React, Node.js, Express, Django
Tools: Git, Docker, AWS, Jenkins
Databases: PostgreSQL, MongoDB
"""
        else:  # Excellent
            sample_resume = """
JOHN DOE
Email: john@example.com | Phone: (123) 456-7890 | LinkedIn: linkedin.com/in/johndoe
Portfolio: johndoe.dev | GitHub: github.com/johndoe

PROFESSIONAL SUMMARY
Senior Software Engineer with 5+ years of experience specializing in full-stack development,
machine learning integration, and cloud architecture. Proven track record of delivering
high-performance applications that scale to millions of users.

PROFESSIONAL EXPERIENCE

SENIOR SOFTWARE ENGINEER
TechCorp Inc. | 2019 - Present
• Led development of a microservices architecture that improved system reliability by 99.9%
• Implemented CI/CD pipelines reducing deployment time by 75%
• Mentored junior developers and conducted code reviews for team of 8 engineers
• Optimized database queries resulting in 40% performance improvement

SOFTWARE ENGINEER
InnovateSoft | 2017 - 2019
• Developed RESTful APIs using Node.js and Express serving 50,000+ daily users
• Created responsive front-end interfaces with React and Redux
• Implemented automated testing suite achieving 90% code coverage

EDUCATION

MASTER OF COMPUTER SCIENCE
Stanford University | 2015 - 2017
• GPA: 3.9/4.0
• Specialization in Machine Learning and Artificial Intelligence

BACHELOR OF SCIENCE IN COMPUTER ENGINEERING
MIT | 2011 - 2015
• GPA: 3.8/4.0
• Dean's List all semesters

TECHNICAL SKILLS

• Languages: Python, JavaScript, TypeScript, Java, SQL, HTML/CSS
• Frameworks: React, Node.js, Express, Django, Flask, TensorFlow
• Tools: Docker, Kubernetes, AWS, Git, Jenkins, Jira
• Databases: PostgreSQL, MongoDB, Redis, Elasticsearch

PROJECTS

INTELLIGENT CUSTOMER SEGMENTATION SYSTEM
• Developed machine learning model that increased marketing conversion rates by 35%
• Implemented data pipeline processing 10TB of customer data daily

DISTRIBUTED CLOUD MONITORING SOLUTION
• Created scalable monitoring system for cloud infrastructure
• Reduced alert response time by 60% through intelligent notification routing
"""
        
        resume_text = st.text_area("Enter or edit the resume text", sample_resume, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        if st.button("Analyze Resume", type="primary"):
            with st.spinner("Analyzing resume..."):
                # Initialize the ML analyzer
                analyzer = MLResumeAnalyzer()
                
                # Analyze the resume
                start_time = time.time()
                results = analyzer.analyze_resume(resume_text, job_description)
                analysis_time = time.time() - start_time
                
                # Display results
                st.markdown('<h3 class="section-header">Analysis Results</h3>', unsafe_allow_html=True)
                
                # Metrics
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                
                # ATS Score
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">ATS Score</div>
                    <div class="metric-value" style="color: {'#4CAF50' if results['ats_score'] >= 80 else '#FFA500' if results['ats_score'] >= 60 else '#F44336'}">{results['ats_score']}%</div>
                    <div>{"Excellent" if results['ats_score'] >= 80 else "Good" if results['ats_score'] >= 60 else "Needs Improvement"}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Format Score
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Format Score</div>
                    <div class="metric-value" style="color: {'#4CAF50' if results['format_score'] >= 80 else '#FFA500' if results['format_score'] >= 60 else '#F44336'}">{results['format_score']}%</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Section Score
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Section Score</div>
                    <div class="metric-value" style="color: {'#4CAF50' if results['section_score'] >= 80 else '#FFA500' if results['section_score'] >= 60 else '#F44336'}">{results['section_score']}%</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Keyword Match
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Keyword Match</div>
                    <div class="metric-value" style="color: {'#4CAF50' if results['keyword_match']['score'] >= 80 else '#FFA500' if results['keyword_match']['score'] >= 60 else '#F44336'}">{results['keyword_match']['score']}%</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Analysis Time
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Analysis Time</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{analysis_time:.2f}s</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Keyword Match Details
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Keyword Match Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Found Skills")
                    if results['keyword_match']['found_skills']:
                        for skill in results['keyword_match']['found_skills']:
                            st.markdown(f"✅ {skill}")
                    else:
                        st.markdown("No matching skills found")
                
                with col2:
                    st.markdown("#### Missing Skills")
                    if results['keyword_match']['missing_skills']:
                        for skill in results['keyword_match']['missing_skills']:
                            st.markdown(f"❌ {skill}")
                    else:
                        st.markdown("No missing skills!")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Suggestions
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Improvement Suggestions")
                
                for suggestion in results['suggestions']:
                    st.markdown(f'''
                    <div class="suggestion-item">
                        <i class="{suggestion['icon']}"></i> {suggestion['text']}
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Performance Tab
    with tabs[1]:
        st.markdown('<h2 class="section-header">Model Performance</h2>', unsafe_allow_html=True)
        
        # Create sample data for visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Resume Quality vs. Scores")
        
        # Sample data
        qualities = ["Poor", "Average", "Good", "Excellent"]
        ats_scores = [30, 55, 75, 95]
        format_scores = [25, 60, 80, 90]
        section_scores = [20, 50, 70, 85]
        keyword_scores = [15, 45, 65, 90]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Quality': qualities,
            'ATS Score': ats_scores,
            'Format Score': format_scores,
            'Section Score': section_scores,
            'Keyword Match': keyword_scores
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set width of bars
        barWidth = 0.15
        
        
        # Set positions of bars on X axis
        r1 = np.arange(len(qualities))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]
        
        # Create bars
        ax.bar(r1, ats_scores, width=barWidth, label='ATS Score', color='#4CAF50')
        ax.bar(r2, format_scores, width=barWidth, label='Format Score', color='#2196F3')
        ax.bar(r3, section_scores, width=barWidth, label='Section Score', color='#FFC107')
        ax.bar(r4, keyword_scores, width=barWidth, label='Keyword Match', color='#9C27B0')
        
        # Add labels and legend
        ax.set_xlabel('Resume Quality', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Scores by Resume Quality')
        ax.set_xticks([r + barWidth*1.5 for r in range(len(qualities))])
        ax.set_xticklabels(qualities)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Display the plot
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model accuracy metrics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Accuracy Metrics")
        
        # Sample metrics
        metrics = {
            'ATS Score Model': {'MSE': 12.45, 'R²': 0.87, 'Training Time': '3.2s'},
            'Format Score Model': {'MSE': 15.32, 'R²': 0.82, 'Training Time': '2.8s'},
            'Section Score Model': {'MSE': 14.18, 'R²': 0.84, 'Training Time': '2.9s'}
        }
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics).T.reset_index()
        metrics_df.columns = ['Model', 'MSE', 'R²', 'Training Time']
        
        # Display as table
        st.table(metrics_df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset Exploration Tab
    with tabs[2]:
        st.markdown('<h2 class="section-header">Dataset Exploration</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        st.markdown("""
        The model is trained on a synthetic dataset of resume samples with varying quality levels. 
        Each resume is labeled with scores for different aspects:
        
        - **ATS Score**: Overall score for Applicant Tracking System compatibility
        - **Format Score**: Score for resume formatting and structure
        - **Section Score**: Score for inclusion and quality of essential resume sections
        - **Keyword Match**: Score for matching job description keywords
        
        The dataset includes resumes of different quality levels, from poor to excellent, 
        with corresponding scores assigned based on quality factors.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample dataset
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sample Dataset")
        
        # Generate sample data
        np.random.seed(42)
        sample_data = []
        
        for quality in ["Poor", "Average", "Good", "Excellent"]:
            quality_factor = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}[quality]
            
            for _ in range(5):  # 5 samples per quality level
                base_score = quality_factor * 20
                variation = np.random.normal(0, 5)
                
                ats_score = min(100, max(0, base_score + variation))
                format_score = min(100, max(0, base_score + np.random.normal(0, 7)))
                section_score = min(100, max(0, base_score + np.random.normal(0, 6)))
                keyword_score = min(100, max(0, base_score + np.random.normal(0, 8)))
                
                sample_data.append({
                    "Quality": quality,
                    "Word Count": int(150 * quality_factor + np.random.normal(0, 50)),
                    "Skills Count": int(3 * quality_factor + np.random.normal(0, 1)),
                    "ATS Score": round(ats_score, 1),
                    "Format Score": round(format_score, 1),
                    "Section Score": round(section_score, 1),
                    "Keyword Match": round(keyword_score, 1)
                })
        
        # Create DataFrame
        sample_df = pd.DataFrame(sample_data)
        
        # Display sample
        st.dataframe(sample_df)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution of scores
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Score Distributions")
        
        # Create distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distributions
        sns.kdeplot(sample_df['ATS Score'], ax=ax, label='ATS Score', color='#4CAF50')
        sns.kdeplot(sample_df['Format Score'], ax=ax, label='Format Score', color='#2196F3')
        sns.kdeplot(sample_df['Section Score'], ax=ax, label='Section Score', color='#FFC107')
        sns.kdeplot(sample_df['Keyword Match'], ax=ax, label='Keyword Match', color='#9C27B0')
        
        # Add labels and legend
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Scores')
        ax.legend()
        ax.grid(linestyle='--', alpha=0.7)
        
        # Display the plot
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
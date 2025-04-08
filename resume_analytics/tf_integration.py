import os
import logging
from datetime import datetime
from .tf_models import ResumeTFModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tf_integration')

class TFResumeAnalyzer:
    """Integration class that combines traditional resume analysis with TensorFlow-based scoring"""
    
    def __init__(self):
        """Initialize the TensorFlow-enhanced resume analyzer"""
        # Initialize the TensorFlow model
        self.tf_model = ResumeTFModel()
        
        # Check if models exist, if not, train them
        self._ensure_models_exist()
    
    def _ensure_models_exist(self):
        """Ensure that TensorFlow models are available"""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        model_files = [
            os.path.join(model_dir, 'tf_ats_score_model'),
            os.path.join(model_dir, 'tf_format_score_model'),
            os.path.join(model_dir, 'tf_section_score_model'),
            os.path.join(model_dir, 'tfidf_vectorizer.pkl'),
            os.path.join(model_dir, 'tf_feature_scaler.pkl')
        ]
        
        # Check if all model files exist
        if not all(os.path.exists(file) for file in model_files):
            logger.info("Some TensorFlow models are missing. Training new models...")
            self.train_models()
    
    def train_models(self, dataset_path=None):
        """Train TensorFlow models for resume scoring"""
        logger.info("Training TensorFlow models for resume scoring")
        self.tf_model.train_models(dataset_path)
    
    def analyze_resume(self, resume_text, job_description=None):
        """Analyze resume using TensorFlow models and return enhanced scores"""
        # Get TensorFlow-based scores
        tf_scores = self.tf_model.predict_scores(resume_text, job_description)
        
        # Extract required skills from job description if available
        required_skills = []
        if job_description:
            # Simple extraction of skills from job description
            # This could be enhanced with more sophisticated NLP techniques
            common_skills = [
                "python", "java", "javascript", "react", "node.js", "sql",
                "html", "css", "aws", "docker", "kubernetes", "git",
                "machine learning", "ai", "data science", "analytics"
            ]
            required_skills = [skill for skill in common_skills if skill in job_description.lower()]
        
        # Prepare the analysis results
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "ats_score": tf_scores["ats_score"],
            "format_score": tf_scores["format_score"],
            "section_score": tf_scores["section_score"],
            "keyword_match": {
                "score": 0,  # Will be calculated below if required_skills is not empty
                "found_skills": [],
                "missing_skills": []
            },
            "suggestions": []
        }
        
        # Calculate keyword match if required skills are available
        if required_skills:
            found_skills = []
            missing_skills = []
            
            for skill in required_skills:
                if skill in resume_text.lower():
                    found_skills.append(skill)
                else:
                    missing_skills.append(skill)
            
            match_score = (len(found_skills) / len(required_skills)) * 100 if required_skills else 0
            
            analysis_results["keyword_match"] = {
                "score": round(match_score),
                "found_skills": found_skills,
                "missing_skills": missing_skills
            }
        
        # Generate suggestions based on scores
        suggestions = []
        
        # ATS score suggestions
        if analysis_results["ats_score"] < 70:
            suggestions.append({
                "icon": "fa-file-alt",
                "text": "Your resume needs improvement to pass ATS systems. Consider reformatting and adding more relevant keywords."
            })
        
        # Format score suggestions
        if analysis_results["format_score"] < 70:
            suggestions.append({
                "icon": "fa-list",
                "text": "Improve your resume format with clear section headers, bullet points, and consistent spacing."
            })
        
        # Section score suggestions
        if analysis_results["section_score"] < 70:
            suggestions.append({
                "icon": "fa-puzzle-piece",
                "text": "Ensure your resume includes all essential sections: Contact Information, Summary, Experience, Education, and Skills."
            })
        
        # Keyword match suggestions
        if analysis_results["keyword_match"]["score"] < 70 and required_skills:
            missing = ", ".join(analysis_results["keyword_match"]["missing_skills"][:3])
            if missing:
                suggestions.append({
                    "icon": "fa-key",
                    "text": f"Add missing keywords to match job requirements: {missing}" + 
                           (" and more." if len(analysis_results["keyword_match"]["missing_skills"]) > 3 else ".")
                })
        
        # Add a positive suggestion if scores are good
        if not suggestions:
            suggestions.append({
                "icon": "fa-star",
                "text": "Your resume is well-optimized! Consider adding more quantifiable achievements to stand out further."
            })
        
        analysis_results["suggestions"] = suggestions
        
        return analysis_results

# Example usage
if __name__ == "__main__":
    # Create the TensorFlow-enhanced analyzer
    analyzer = TFResumeAnalyzer()
    
    # Sample resume and job description
    sample_resume = """
    CONTACT INFORMATION
    Email: john.doe@example.com
    Phone: (123) 456-7890
    LinkedIn: linkedin.com/in/johndoe
    
    SUMMARY
    Experienced software engineer with 5 years of experience in Python development and machine learning.
    
    EXPERIENCE
    Senior Software Engineer
    ABC Tech | 2018 - Present
    • Developed machine learning models for customer segmentation
    • Led a team of 5 developers on a major product launch
    • Improved system performance by 40%
    
    Software Engineer
    XYZ Corp | 2015 - 2018
    • Built RESTful APIs using Django
    • Implemented CI/CD pipelines
    
    EDUCATION
    Master of Computer Science
    University of Technology | 2013 - 2015
    
    Bachelor of Computer Science
    State University | 2009 - 2013
    
    SKILLS
    Python, Java, Machine Learning, SQL, Django, Flask, Docker, Kubernetes, Git
    """
    
    sample_job = """Looking for a Senior Software Engineer with experience in Python, Machine Learning, and team leadership."""
    
    # Analyze the resume
    results = analyzer.analyze_resume(sample_resume, sample_job)
    
    # Print the results
    print(f"ATS Score: {results['ats_score']}")
    print(f"Format Score: {results['format_score']}")
    print(f"Section Score: {results['section_score']}")
    print(f"Keyword Match: {results['keyword_match']['score']}%")
    print("\nSuggestions:")
    for suggestion in results['suggestions']:
        print(f"- {suggestion['text']}")
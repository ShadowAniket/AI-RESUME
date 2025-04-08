import os
import logging
from datetime import datetime
from .bert_tf_models import ResumeBERTTFModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bert_tf_integration')

class BERTTFResumeAnalyzer:
    """Integration class that combines traditional resume analysis with BERT-TensorFlow-based scoring"""
    
    def __init__(self):
        """Initialize the BERT-TensorFlow-enhanced resume analyzer"""
        # Initialize the BERT-TensorFlow model
        self.bert_tf_model = ResumeBERTTFModel()
        
        # Check if models exist, if not, train them
        self._ensure_models_exist()
    
    def _ensure_models_exist(self):
        """Ensure that BERT-TensorFlow models are available"""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        model_files = [
            os.path.join(model_dir, 'bert_tf_ats_score_model'),
            os.path.join(model_dir, 'bert_tf_format_score_model'),
            os.path.join(model_dir, 'bert_tf_section_score_model'),
            os.path.join(model_dir, 'bert_tf_feature_scaler.pkl')
        ]
        
        # Check if all model files exist
        if not all(os.path.exists(file) for file in model_files):
            logger.info("Some BERT-TensorFlow models are missing. Training new models...")
            self.train_models()
    
    def train_models(self, dataset_path=None):
        """Train BERT-TensorFlow models for resume scoring"""
        logger.info("Training BERT-TensorFlow models for resume scoring")
        try:
            # Add validation check for dataset path
            if dataset_path and not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
                
            # Train models with progress tracking
            training_history = self.bert_tf_model.train_models(dataset_path)
            
            # Validate training results
            if not training_history or 'loss' not in training_history.history:
                raise RuntimeError("Model training failed - no valid training history returned")
                
            logger.info("BERT-TensorFlow models trained successfully")
            return training_history
            
        except Exception as e:
            logger.error(f"Error during BERT-TensorFlow model training: {str(e)}")
            # Clean up potentially corrupted model files
            self._cleanup_partial_models()
            raise
    
    def analyze_resume(self, resume_text, job_description=None):
        """Analyze resume using BERT-TensorFlow models and return enhanced scores"""
        # Get BERT-TensorFlow-based scores
        bert_tf_scores = self.bert_tf_model.predict_scores(resume_text, job_description)
        
        # Extract required skills from job description if available
        required_skills = []
        if job_description:
            # Enhanced skill extraction with focus on TensorFlow, Keras, and BERT
            common_skills = [
                "python", "java", "javascript", "react", "node.js", "sql",
                "html", "css", "aws", "docker", "kubernetes", "git",
                "machine learning", "ai", "data science", "analytics",
                "tensorflow", "keras", "bert", "deep learning", "neural networks",
                "nlp", "natural language processing", "computer vision"
            ]
            required_skills = [skill for skill in common_skills if skill in job_description.lower()]
        
        # Prepare the analysis results
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "ats_score": bert_tf_scores["ats_score"],
            "format_score": bert_tf_scores["format_score"],
            "section_score": bert_tf_scores["section_score"],
            "keyword_match": {
                "score": 0,  # Will be calculated below if required_skills is not empty
                "found_skills": [],
                "missing_skills": []
            },
            "suggestions": bert_tf_scores.get("suggestions", [])
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
        
        return analysis_results
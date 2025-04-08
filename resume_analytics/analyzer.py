import spacy
from collections import Counter
from datetime import datetime
import os
import logging

# Import our integration modules
try:
    from .ml_integration import MLResumeAnalyzer
except ImportError:
    MLResumeAnalyzer = None

try:
    from .tf_integration import TFResumeAnalyzer
except ImportError:
    TFResumeAnalyzer = None
    
try:
    from .bert_tf_integration import BERTTFResumeAnalyzer
except ImportError:
    BERTTFResumeAnalyzer = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('resume_analyzer')

class ResumeAnalyzer:
    def __init__(self, model_type='bert_tf'):
        """Initialize the resume analyzer
        
        Args:
            model_type: Type of model to use ('ml' for scikit-learn, 'tf' for TensorFlow, 'bert_tf' for BERT-TensorFlow)
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.model_type = model_type
        
        # Initialize the BERT-TensorFlow-enhanced analyzer (preferred)
        self.bert_tf_enabled = False
        if model_type == 'bert_tf':
            try:
                if BERTTFResumeAnalyzer:
                    self.bert_tf_analyzer = BERTTFResumeAnalyzer()
                    self.bert_tf_enabled = True
                    logger.info("BERT-TensorFlow-enhanced resume analysis enabled")
                else:
                    logger.warning("BERT-TensorFlow integration module not available")
            except Exception as e:
                logger.warning(f"BERT-TensorFlow-enhanced resume analysis disabled: {str(e)}")
        
        # Initialize the TensorFlow-enhanced analyzer (second preference)
        self.tf_enabled = False
        if not self.bert_tf_enabled and model_type in ['tf', 'tensorflow']:
            try:
                if TFResumeAnalyzer:
                    self.tf_analyzer = TFResumeAnalyzer()
                    self.tf_enabled = True
                    logger.info("TensorFlow-enhanced resume analysis enabled")
                else:
                    logger.warning("TensorFlow integration module not available")
            except Exception as e:
                logger.warning(f"TensorFlow-enhanced resume analysis disabled: {str(e)}")
        
        # Initialize the ML-enhanced analyzer as fallback
        self.ml_enabled = False
        if not self.bert_tf_enabled and not self.tf_enabled or model_type == 'ml':
            try:
                if MLResumeAnalyzer:
                    self.ml_analyzer = MLResumeAnalyzer()
                    self.ml_enabled = True
                    logger.info("ML-enhanced resume analysis enabled")
                else:
                    logger.warning("ML integration module not available")
            except Exception as e:
                logger.warning(f"ML-enhanced resume analysis disabled: {str(e)}")
                
        if not self.bert_tf_enabled and not self.tf_enabled and not self.ml_enabled:
            logger.warning("No ML, TensorFlow, or BERT-TensorFlow models available. Using basic analysis only.")
        
    def analyze_resume(self, resume_data, job_role_info=None):
        """Analyze resume text and return metrics
        
        Args:
            resume_data: Either a string containing resume text or a dict with 'raw_text' key
            job_role_info: Optional dict containing job role information
        
        Returns:
            Dict containing analysis results
        """
        # Extract text from input
        if isinstance(resume_data, dict) and 'raw_text' in resume_data:
            resume_text = resume_data['raw_text']
        else:
            resume_text = resume_data
        
        # Process with spaCy
        doc = self.nlp(resume_text)
        
        # Basic metrics
        word_count = len(resume_text.split())
        sentence_count = len(list(doc.sents))
        
        # Skills extraction
        skills = self._extract_skills(doc)
        
        # Experience analysis
        experience_years = self._analyze_experience(doc)
        
        # Extract job description if available
        job_description = None
        required_skills = []
        if job_role_info:
            job_description = job_role_info.get('description', '')
            required_skills = job_role_info.get('required_skills', [])
        
        # Use BERT-TensorFlow-enhanced analysis if available and enabled
        if self.bert_tf_enabled and self.model_type == 'bert_tf':
            try:
                bert_tf_results = self.bert_tf_analyzer.analyze_resume(resume_text, job_description)
                
                # Combine traditional and BERT-TensorFlow-based results
                analysis_results = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "skills_count": len(skills),
                        "experience_years": experience_years,
                        "profile_score": bert_tf_results["ats_score"]  # Use BERT-TensorFlow-based ATS score as profile score
                    },
                    "skills": list(skills),
                    "ats_score": bert_tf_results["ats_score"],
                    "format_score": bert_tf_results["format_score"],
                    "section_score": bert_tf_results["section_score"],
                    "keyword_match": bert_tf_results["keyword_match"],
                    "suggestions": bert_tf_results["suggestions"],
                    "model_type": "bert-tensorflow"
                }
                
                return analysis_results
            except Exception as e:
                logger.error(f"Error in BERT-TensorFlow-based analysis: {str(e)}")
                logger.info("Falling back to TensorFlow-based, ML-based, or traditional analysis")
        
        # Use TensorFlow-enhanced analysis if available and enabled
        if self.tf_enabled and (self.model_type == 'tf' or self.model_type == 'tensorflow'):
            try:
                tf_results = self.tf_analyzer.analyze_resume(resume_text, job_description)
                
                # Combine traditional and TensorFlow-based results
                analysis_results = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "skills_count": len(skills),
                        "experience_years": experience_years,
                        "profile_score": tf_results["ats_score"]  # Use TensorFlow-based ATS score as profile score
                    },
                    "skills": list(skills),
                    "ats_score": tf_results["ats_score"],
                    "format_score": tf_results["format_score"],
                    "section_score": tf_results["section_score"],
                    "keyword_match": tf_results["keyword_match"],
                    "suggestions": tf_results["suggestions"],
                    "model_type": "tensorflow"
                }
                
                return analysis_results
            except Exception as e:
                logger.error(f"Error in TensorFlow-based analysis: {str(e)}")
                logger.info("Falling back to ML-based or traditional analysis")
        
        # Use ML-enhanced analysis if available or if TensorFlow failed
        if self.ml_enabled:
            try:
                ml_results = self.ml_analyzer.analyze_resume(resume_text, job_description)
                
                # Combine traditional and ML-based results
                analysis_results = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "skills_count": len(skills),
                        "experience_years": experience_years,
                        "profile_score": ml_results["ats_score"]  # Use ML-based ATS score as profile score
                    },
                    "skills": list(skills),
                    "ats_score": ml_results["ats_score"],
                    "format_score": ml_results["format_score"],
                    "section_score": ml_results["section_score"],
                    "keyword_match": ml_results["keyword_match"],
                    "suggestions": ml_results["suggestions"],
                    "model_type": "scikit-learn"
                }
                
                return analysis_results
            except Exception as e:
                logger.error(f"Error in ML-based analysis: {str(e)}")
                logger.info("Falling back to traditional analysis")
        
        # Calculate profile score using traditional method
        profile_score = self._calculate_profile_score(
            word_count, sentence_count, len(skills), experience_years
        )
        
        # Calculate keyword match if required skills are provided
        keyword_match = {
            "score": 0,
            "found_skills": [],
            "missing_skills": []
        }
        
        if required_skills:
            found_skills = []
            missing_skills = []
            
            for skill in required_skills:
                if skill.lower() in resume_text.lower() or any(skill.lower() in s.lower() for s in skills):
                    found_skills.append(skill)
                else:
                    missing_skills.append(skill)
            
            match_score = (len(found_skills) / len(required_skills)) * 100 if required_skills else 0
            
            keyword_match = {
                "score": round(match_score),
                "found_skills": found_skills,
                "missing_skills": missing_skills
            }
        
        # Generate suggestions using traditional method
        suggestions = self._generate_suggestions(
            word_count, sentence_count, skills, experience_years, keyword_match
        )
        
        # Estimate format and section scores based on traditional metrics
        format_score = min(100, max(0, profile_score + 10))
        section_score = min(100, max(0, profile_score))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "skills_count": len(skills),
                "experience_years": experience_years,
                "profile_score": profile_score
            },
            "skills": list(skills),
            "ats_score": profile_score,  # Use profile score as ATS score in traditional analysis
            "format_score": format_score,
            "section_score": section_score,
            "keyword_match": keyword_match,
            "suggestions": suggestions
        }
    
    def _extract_skills(self, doc):
        """Extract skills from resume"""
        # Common technical skills keywords
        tech_skills = {
            "python", "java", "javascript", "react", "node.js", "sql",
            "html", "css", "aws", "docker", "kubernetes", "git",
            "machine learning", "ai", "data science", "analytics",
            "c++", "c#", "ruby", "php", "swift", "kotlin", "go",
            "django", "flask", "spring", "express", "angular", "vue",
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
            "mongodb", "postgresql", "mysql", "oracle", "redis",
            "rest api", "graphql", "microservices", "devops", "ci/cd",
            "agile", "scrum", "jira", "git", "github", "gitlab",
            "cloud", "aws", "azure", "gcp", "docker", "kubernetes"
        }
        
        skills = set()
        for token in doc:
            if token.text.lower() in tech_skills:
                skills.add(token.text)
            # Check for compound skills (e.g., "machine learning")
            if token.i < len(doc) - 1:
                bigram = (token.text + " " + doc[token.i + 1].text).lower()
                if bigram in tech_skills:
                    skills.add(bigram)
        
        return skills
    
    def _analyze_experience(self, doc):
        """Analyze years of experience"""
        # Simple heuristic - look for number + "years"
        experience_years = 0
        for token in doc:
            if token.like_num and token.i < len(doc) - 1:
                next_token = doc[token.i + 1]
                if "year" in next_token.text.lower():
                    try:
                        experience_years = max(experience_years, int(token.text))
                    except ValueError:
                        continue
        return experience_years
    
    def _calculate_profile_score(self, word_count, sentence_count, skills_count, experience_years):
        """Calculate profile score based on various metrics"""
        score = 0
        
        # Word count scoring (0-25 points)
        if word_count >= 300:
            score += 25
        else:
            score += (word_count / 300) * 25
        
        # Skills scoring (0-35 points)
        if skills_count >= 8:
            score += 35
        else:
            score += (skills_count / 8) * 35
        
        # Experience scoring (0-40 points)
        if experience_years >= 5:
            score += 40
        else:
            score += (experience_years / 5) * 40
        
        return min(round(score), 100)
    
    def _generate_suggestions(self, word_count, sentence_count, skills, experience_years, keyword_match=None):
        """Generate improvement suggestions based on analysis"""
        suggestions = []
        
        if word_count < 300:
            suggestions.append({
                "icon": "fa-file-text",
                "text": "Add more detail to your resume - aim for at least 300 words"
            })
            
        if len(skills) < 8:
            suggestions.append({
                "icon": "fa-code",
                "text": "Include more relevant technical skills and technologies"
            })
            
        if sentence_count < 10:
            suggestions.append({
                "icon": "fa-list",
                "text": "Add more achievements and responsibilities from your experience"
            })
            
        if experience_years < 2:
            suggestions.append({
                "icon": "fa-briefcase",
                "text": "Highlight any internships, projects, or relevant coursework"
            })
        
        # Add keyword match suggestions if available
        if keyword_match and keyword_match["score"] < 70 and keyword_match["missing_skills"]:
            missing = ", ".join(keyword_match["missing_skills"][:3])
            if missing:
                suggestions.append({
                    "icon": "fa-key",
                    "text": f"Add missing keywords to match job requirements: {missing}" + 
                           (" and more." if len(keyword_match["missing_skills"]) > 3 else ".")
                })
            
        if not suggestions:
            suggestions.append({
                "icon": "fa-star",
                "text": "Your resume looks great! Consider adding more quantifiable achievements"
            })
            
        return suggestions

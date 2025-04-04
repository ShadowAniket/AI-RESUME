import spacy
from collections import Counter
from datetime import datetime

class ResumeAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def analyze_resume(self, resume_text):
        """Analyze resume text and return metrics"""
        doc = self.nlp(resume_text)
        
        # Basic metrics
        word_count = len(resume_text.split())
        sentence_count = len(list(doc.sents))
        
        # Skills extraction
        skills = self._extract_skills(doc)
        
        # Experience analysis
        experience_years = self._analyze_experience(doc)
        
        # Calculate profile score
        profile_score = self._calculate_profile_score(
            word_count, sentence_count, len(skills), experience_years
        )
        
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
            "suggestions": self._generate_suggestions(
                word_count, sentence_count, skills, experience_years
            )
        }
    
    def _extract_skills(self, doc):
        """Extract skills from resume"""
        # Common technical skills keywords
        tech_skills = {
            "python", "java", "javascript", "react", "node.js", "sql",
            "html", "css", "aws", "docker", "kubernetes", "git",
            "machine learning", "ai", "data science", "analytics"
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
    
    def _generate_suggestions(self, word_count, sentence_count, skills, experience_years):
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
            
        if not suggestions:
            suggestions.append({
                "icon": "fa-star",
                "text": "Your resume looks great! Consider adding more quantifiable achievements"
            })
            
        return suggestions

import PyPDF2
import docx
import re
from io import BytesIO

class ResumeParser:
    def __init__(self):
        pass
        
    def extract_text_from_pdf(self, pdf_file):
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
            
    def extract_text_from_docx(self, docx_file):
        try:
            doc = docx.Document(BytesIO(docx_file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""
            
    def extract_text(self, file):
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        if file.name.endswith('.pdf'):
            return self.extract_text_from_pdf(file)
        elif file.name.endswith('.docx'):
            return self.extract_text_from_docx(file)
        else:
            return ""
            
    def parse(self, file):
        text = self.extract_text(file)
        
        # Simple keyword-based parsing
        skills = []
        experience = []
        education = []
        
        # Common programming languages and tools
        skill_keywords = ['python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'angular', 'vue', 
                         'node', 'express', 'django', 'flask', 'spring', 'docker', 'kubernetes', 'aws', 
                         'azure', 'git', 'jenkins', 'jira']
                         
        # Look for skills
        text_lower = text.lower()
        for skill in skill_keywords:
            if skill in text_lower:
                skills.append(skill)
                
        return {
            "skills": skills,
            "experience": experience,
            "education": education,
            "raw_text": text
        }

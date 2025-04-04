from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Create the base class for declarative models
Base = declarative_base()

# Define the Resume model
class Resume(Base):
    __tablename__ = 'resumes'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100))
    job_role = Column(String(100))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

# Define the Analysis model
class Analysis(Base):
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer)
    analysis_data = Column(Text)  # Store JSON data
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DatabaseManager:
    def __init__(self, db_path='resume_data.db'):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_resume(self, user_id, job_role, content):
        resume = Resume(
            user_id=user_id,
            job_role=job_role,
            content=content
        )
        self.session.add(resume)
        self.session.commit()
        return resume.id
    
    def get_resume(self, resume_id):
        return self.session.query(Resume).filter(Resume.id == resume_id).first()
    
    def get_user_resumes(self, user_id):
        return self.session.query(Resume).filter(Resume.user_id == user_id).all()
    
    def save_analysis(self, resume_id, analysis_data):
        analysis = Analysis(
            resume_id=resume_id,
            analysis_data=analysis_data
        )
        self.session.add(analysis)
        self.session.commit()
        return analysis.id
    
    def get_analysis(self, analysis_id):
        return self.session.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    def get_resume_analyses(self, resume_id):
        return self.session.query(Analysis).filter(Analysis.resume_id == resume_id).all()
    
    def close(self):
        self.session.close()

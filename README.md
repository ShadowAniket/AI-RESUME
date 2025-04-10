# Smart Resume AI 🚀

A powerful AI-driven platform for optimizing resumes and advancing careers, built with Python and Streamlit.

## Features

### 🔍 AI-Powered Resume Analysis
- Instant AI feedback on resume content and format
- ATS (Applicant Tracking System) compatibility scoring
- Keyword matching against job requirements
- Section-by-section analysis and recommendations
- Skills gap identification

### 📝 Smart Resume Builder
- Professional resume templates (Modern, Professional, Minimal, Creative)
- Intelligent content suggestions
- Real-time formatting
- Section-based organization (Personal Info, Experience, Education, Skills, Projects)
- Export to DOCX format

### 📊 Analytics Dashboard
- Resume performance metrics
- Skills analysis
- Industry insights
- Historical data tracking

### 🎯 Job Search Integration
- Tailored job recommendations
- Job portal integration
- Role-specific skill requirements
- Customized learning resources

### 💬 Feedback System
- User feedback collection
- Feature requests
- Analytics and insights
- Continuous improvement

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI/ML**: 
  - Natural Language Processing
  - Machine Learning Models
  - BERT/TensorFlow Integration
- **Database**: SQLite
- **File Handling**: PDF, DOCX support

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd AI-RESUME
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Initialize the database
```bash
python init_default_admin.py
```

4. Run the application
```bash
streamlit run app.py
```

## Project Structure

```
AI-RESUME/
├── app.py                    # Main application file
├── ui_components.py         # UI component definitions
├── requirements.txt         # Project dependencies
├── resume_analytics/        # AI/ML analysis modules
├── dashboard/              # Analytics dashboard
├── feedback/              # User feedback system
├── jobs/                  # Job search integration
├── utils/                # Utility functions
├── config/              # Configuration files
└── assets/             # Static assets
```

## Features in Detail

### Resume Analysis
- Analyzes resume content and structure
- Provides ATS compatibility score
- Identifies missing keywords and skills
- Suggests improvements for each section
- Formatting recommendations

### Resume Builder
- Multiple professional templates
- Section-based organization
- Skills categorization (Technical, Soft, Languages, Tools)
- Project and experience formatting
- Education and certification sections

### Analytics
- Resume performance tracking
- Skills gap analysis
- Industry trends
- Success metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
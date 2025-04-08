# Resume Analytics ML Module

This module enhances the AI-RESUME application with machine learning capabilities for more accurate resume scoring and analysis.

## Features

- **ML-based Resume Scoring**: Uses machine learning models to provide accurate ATS compatibility scores
- **Format Analysis**: Evaluates resume formatting and structure
- **Section Analysis**: Assesses the quality and completeness of resume sections
- **Keyword Matching**: Analyzes resume against job descriptions for keyword compatibility
- **Personalized Suggestions**: Provides tailored improvement recommendations

## Components

- `ml_models.py`: Core ML model implementation for resume scoring
- `ml_integration.py`: Integration layer between ML models and existing analyzer
- `train_test_models.py`: Script for training and testing ML models
- `ml_demo.py`: Streamlit-based demo application for the ML models
- `analyzer.py`: Updated analyzer with ML capabilities

## Getting Started

### Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

### Training the Models

To train the ML models with synthetic data:

```bash
python -m resume_analytics.train_test_models
```

This will:
1. Generate a synthetic dataset of resumes
2. Train the ML models for ATS scoring, format analysis, and section analysis
3. Evaluate the models on test samples
4. Save the trained models to the `resume_analytics/models` directory

### Running the Demo

To run the Streamlit demo application:

```bash
streamlit run resume_analytics/ml_demo.py
```

This will launch a web interface where you can:
- Analyze resumes with ML-based scoring
- View model performance metrics
- Explore the training dataset

## Integration with Existing Application

The ML capabilities are automatically integrated with the existing resume analyzer. When a resume is analyzed, the system will:

1. First attempt to use the ML-based scoring
2. Fall back to traditional scoring methods if ML is unavailable

No changes to the existing API are required - the enhanced functionality is available through the same interface.

## Model Details

### ATS Score Model

Predicts the overall compatibility of a resume with Applicant Tracking Systems based on:
- Resume content and structure
- Keyword relevance to job descriptions
- Section organization and completeness

### Format Score Model

Evaluates the formatting quality of a resume, including:
- Section headers and organization
- Use of bullet points
- Spacing and layout consistency

### Section Score Model

Assesses the completeness and quality of essential resume sections:
- Contact information
- Professional summary
- Work experience
- Education
- Skills

## Dataset

The models are trained on a synthetic dataset of resumes with varying quality levels. Each resume is labeled with scores for different aspects:

- ATS Score: Overall score for Applicant Tracking System compatibility
- Format Score: Score for resume formatting and structure
- Section Score: Score for inclusion and quality of essential resume sections

The synthetic data generation ensures a diverse range of resume qualities and formats for robust model training.
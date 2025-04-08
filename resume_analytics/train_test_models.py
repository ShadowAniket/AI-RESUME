import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from datetime import datetime

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our ML modules
from resume_analytics.ml_models import ResumeMLModel, download_resume_dataset
from resume_analytics.tf_models import ResumeTFModel, download_tf_resume_dataset
from resume_analytics.bert_tf_models import ResumeBERTTFModel
from resume_analytics.ml_integration import MLResumeAnalyzer
from resume_analytics.tf_integration import TFResumeAnalyzer
from resume_analytics.bert_tf_integration import BERTTFResumeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_test_models')

def train_and_evaluate_models(dataset_path=None, model_type="ml"):
    """Train and evaluate models for resume scoring
    
    Args:
        dataset_path: Path to the dataset file
        model_type: Type of model to use ('ml' for scikit-learn, 'tf' for TensorFlow, 'bert_tf' for BERT-TensorFlow)
    """
    logger.info(f"Starting {model_type.upper()} model training and evaluation")
    start_time = time.time()
    
    # Create the appropriate model based on model_type
    if model_type.lower() == "bert_tf":
        model = ResumeBERTTFModel()
        # Download or generate dataset if not provided
        if not dataset_path:
            logger.info("No dataset provided. Using synthetic BERT-TensorFlow dataset...")
            # Use the same dataset function as TF for now, can be specialized later
            dataset_path = download_tf_resume_dataset('bert_tf_resume_dataset.csv')
    elif model_type.lower() == "tf":
        model = ResumeTFModel()
        # Download or generate dataset if not provided
        if not dataset_path:
            logger.info("No dataset provided. Downloading/generating synthetic TensorFlow dataset...")
            dataset_path = download_tf_resume_dataset('tf_resume_dataset.csv')
    else:  # Default to ML model
        model = ResumeMLModel()
        # Download or generate dataset if not provided
        if not dataset_path:
            logger.info("No dataset provided. Downloading/generating synthetic dataset...")
            dataset_path = download_resume_dataset('resume_dataset.csv')
    
    # Load the dataset
    try:
        data = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset from {dataset_path} with {len(data)} samples")
        
        # Display dataset statistics
        logger.info("Dataset statistics:")
        logger.info(f"ATS Score - Mean: {data['ats_score'].mean():.2f}, Std: {data['ats_score'].std():.2f}")
        logger.info(f"Format Score - Mean: {data['format_score'].mean():.2f}, Std: {data['format_score'].std():.2f}")
        logger.info(f"Section Score - Mean: {data['section_score'].mean():.2f}, Std: {data['section_score'].std():.2f}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return
    
    # Train the models
    logger.info("Training models...")
    model.train_models(dataset_path)
    
    # Evaluate on test samples
    logger.info("Evaluating models on test samples...")
    
    # Create test samples with varying quality
    test_samples = [
        {
            "quality": "Poor",
            "resume": """
            John Doe
            john@example.com
            
            I am looking for a job in software development.
            
            Worked at Company A for 2 years
            Graduated from University B
            
            Skills: Python
            """
        },
        {
            "quality": "Average",
            "resume": """
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
        },
        {
            "quality": "Excellent",
            "resume": """
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
        }
    ]
    
    # Sample job description
    job_description = """
    Senior Software Engineer
    
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
    """
    
    # Evaluate each test sample
    results = []
    for sample in test_samples:
        scores = model.predict_scores(sample["resume"], job_description)
        results.append({
            "quality": sample["quality"],
            "ats_score": scores["ats_score"],
            "format_score": scores["format_score"],
            "section_score": scores["section_score"]
        })
    
    # Display results
    results_df = pd.DataFrame(results)
    logger.info("\nTest Results:")
    logger.info(results_df.to_string(index=False))
    
    # Create a bar chart of the results
    plt.figure(figsize=(12, 6))
    
    # Set up the data for plotting
    qualities = results_df['quality']
    ats_scores = results_df['ats_score']
    format_scores = results_df['format_score']
    section_scores = results_df['section_score']
    
    # Save results to CSV in the results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'{model_type}_test_results_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)
    logger.info(f"Test results saved to {results_file}")
    
    # Set the positions and width for the bars
    pos = np.arange(len(qualities))
    width = 0.25  # Width of the bars

    # Plot the bars
    plt.bar(pos - width, ats_scores, width, alpha=0.7, color='#4CAF50', label='ATS Score')
    plt.bar(pos, format_scores, width, alpha=0.7, color='#2196F3', label='Format Score')
    plt.bar(pos + width, section_scores, width, alpha=0.7, color='#FFC107', label='Section Score')

    # Add labels, title, and legend
    plt.xlabel('Resume Quality')
    plt.ylabel('Score')
    plt.title(f'{model_type.upper()} Model Resume Scoring Results')
    plt.xticks(pos, qualities)
    plt.legend(loc='best')
    plt.ylim(0, 100)

    # Add value labels on top of bars
    for i, v in enumerate(ats_scores):
        plt.text(i - width, v + 2, str(v), ha='center')
    for i, v in enumerate(format_scores):
        plt.text(i, v + 2, str(v), ha='center')
    for i, v in enumerate(section_scores):
        plt.text(i + width, v + 2, str(v), ha='center')

    # Save the plot to the results directory
    plot_file = os.path.join(results_dir, f'{model_type}_resume_scores_{timestamp}.png')
    plt.savefig(plot_file)
    logger.info(f"Plot saved to {plot_file}")

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Calculate and display training time
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training and evaluation completed in {training_time:.2f} seconds")
    
    # Test the integration with appropriate analyzer based on model type
    if model_type.lower() == "bert_tf" and BERTTFResumeAnalyzer:
        logger.info("\nTesting integration with BERTTFResumeAnalyzer...")
        analyzer = BERTTFResumeAnalyzer()
        analysis = analyzer.analyze_resume(test_samples[2]["resume"], job_description)
        
        logger.info("BERTTFResumeAnalyzer Results:")
        logger.info(f"ATS Score: {analysis['ats_score']}")
        logger.info(f"Format Score: {analysis['format_score']}")
        logger.info(f"Section Score: {analysis['section_score']}")
        logger.info(f"Keyword Match: {analysis['keyword_match']['score']}%")
        logger.info("\nSuggestions:")
        for suggestion in analysis['suggestions']:
            logger.info(f"- {suggestion['text']}")
    elif model_type.lower() == "tf" and TFResumeAnalyzer:
        logger.info("\nTesting integration with TFResumeAnalyzer...")
        analyzer = TFResumeAnalyzer()
        analysis = analyzer.analyze_resume(test_samples[2]["resume"], job_description)
        
        logger.info("TFResumeAnalyzer Results:")
        logger.info(f"ATS Score: {analysis['ats_score']}")
        logger.info(f"Format Score: {analysis['format_score']}")
        logger.info(f"Section Score: {analysis['section_score']}")
        logger.info(f"Keyword Match: {analysis['keyword_match']['score']}%")
        logger.info("\nSuggestions:")
        for suggestion in analysis['suggestions']:
            logger.info(f"- {suggestion['text']}")
    elif MLResumeAnalyzer:
        logger.info("\nTesting integration with MLResumeAnalyzer...")
        analyzer = MLResumeAnalyzer()
        analysis = analyzer.analyze_resume(test_samples[2]["resume"], job_description)
        
        logger.info("MLResumeAnalyzer Results:")
        logger.info(f"ATS Score: {analysis['ats_score']}")
        logger.info(f"Format Score: {analysis['format_score']}")
        logger.info(f"Section Score: {analysis['section_score']}")
        logger.info(f"Keyword Match: {analysis['keyword_match']['score']}%")
        logger.info("\nSuggestions:")
        for suggestion in analysis['suggestions']:
            logger.info(f"- {suggestion['text']}")
    else:
        logger.warning("No resume analyzer available for testing")
    
    # Calculate total time
    elapsed_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    

def train_and_evaluate_tf_models(dataset_path=None):
    """Train and evaluate TensorFlow models for resume scoring"""
    return train_and_evaluate_models(dataset_path, model_type='tf')

def train_and_evaluate_bert_tf_models(dataset_path=None):
    """Train and evaluate BERT-TensorFlow models for resume scoring"""
    return train_and_evaluate_models(dataset_path, model_type='bert_tf')

def main():
    """Main function to run the training and evaluation"""
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate resume scoring models')
    parser.add_argument('--model', type=str, choices=['ml', 'tf', 'bert_tf'], default='bert_tf',
                        help='Model type to use: ml (scikit-learn), tf (TensorFlow), or bert_tf (BERT-TensorFlow)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset file (CSV)')
    args = parser.parse_args()
    
    logger.info("Starting resume model training and testing")
    
    # Create output directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Train and evaluate models based on selected type
    if args.model == 'bert_tf':
        logger.info("Using BERT-TensorFlow models for training and evaluation")
        train_and_evaluate_bert_tf_models(args.dataset)
    elif args.model == 'tf':
        logger.info("Using TensorFlow models for training and evaluation")
        train_and_evaluate_tf_models(args.dataset)
    else:
        logger.info("Using scikit-learn models for training and evaluation")
        train_and_evaluate_models(args.dataset, 'ml')
    
    logger.info("Resume model training and testing completed")

if __name__ == "__main__":
    main()
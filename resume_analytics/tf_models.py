import os
import numpy as np
import pandas as pd
import re
import pickle
import joblib
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import spacy
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('resume_tf')

class ResumeTFModel:
    """TensorFlow-based model for resume scoring"""
    
    def __init__(self, model_dir='models'):
        """Initialize the TensorFlow model"""
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize results directory for test results
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Model paths
        self.ats_model_path = os.path.join(self.model_dir, 'tf_ats_score_model')
        self.format_model_path = os.path.join(self.model_dir, 'tf_format_score_model')
        self.section_model_path = os.path.join(self.model_dir, 'tf_section_score_model')
        self.vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        self.scaler_path = os.path.join(self.model_dir, 'tf_feature_scaler.pkl')
        
        # Load models if they exist
        self.ats_model = self._load_tf_model(self.ats_model_path)
        self.format_model = self._load_tf_model(self.format_model_path)
        self.section_model = self._load_tf_model(self.section_model_path)
        self.vectorizer = self._load_joblib_model(self.vectorizer_path)
        self.scaler = self._load_joblib_model(self.scaler_path)
    
    def _load_tf_model(self, model_path):
        """Load a TensorFlow model if it exists"""
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except Exception as e:
                logger.error(f"Error loading TensorFlow model {model_path}: {str(e)}")
                return None
        return None
    
    def _load_joblib_model(self, model_path):
        """Load a joblib model if it exists"""
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                logger.error(f"Error loading joblib model {model_path}: {str(e)}")
                return None
        return None
    
    def _save_tf_model(self, model, model_path):
        """Save a TensorFlow model to disk"""
        try:
            model.save(model_path)
            logger.info(f"TensorFlow model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving TensorFlow model {model_path}: {str(e)}")
            return False
    
    def _save_joblib_model(self, model, model_path):
        """Save a joblib model to disk"""
        try:
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_path}: {str(e)}")
            return False
    
    def _preprocess_text(self, text):
        """Preprocess resume text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lemmatize using spaCy
        doc = self.nlp(text)
        lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop])
        
        return lemmatized
    
    def _extract_features(self, resume_text, job_description=None):
        """Extract features from resume text"""
        # Basic text features
        word_count = len(resume_text.split())
        sentence_count = len(list(self.nlp(resume_text).sents))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Section detection
        sections = {
            'contact': any(keyword in resume_text.lower() for keyword in ['email', 'phone', 'address', 'linkedin']),
            'education': any(keyword in resume_text.lower() for keyword in ['education', 'university', 'college', 'degree']),
            'experience': any(keyword in resume_text.lower() for keyword in ['experience', 'work', 'employment', 'job']),
            'skills': any(keyword in resume_text.lower() for keyword in ['skills', 'technologies', 'tools', 'proficiencies'])
        }
        section_count = sum(sections.values())
        
        # Keyword matching with job description
        keyword_match = 0
        if job_description:
            job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
            resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
            matching_keywords = job_keywords.intersection(resume_words)
            keyword_match = len(matching_keywords) / max(1, len(job_keywords))
        
        # Format features
        has_bullet_points = any(line.strip().startswith(('•', '-', '*')) for line in resume_text.split('\n'))
        has_section_headers = any(line.isupper() for line in resume_text.split('\n'))
        
        # Create feature dictionary
        features = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'section_count': section_count,
            'has_contact': int(sections['contact']),
            'has_education': int(sections['education']),
            'has_experience': int(sections['experience']),
            'has_skills': int(sections['skills']),
            'keyword_match': keyword_match,
            'has_bullet_points': int(has_bullet_points),
            'has_section_headers': int(has_section_headers)
        }
        
        return features
    
    def _create_tf_model(self, input_dim, text_dim):
        """Create a TensorFlow model for resume scoring"""
        # Input layers
        feature_input = Input(shape=(input_dim,), name='feature_input')
        text_input = Input(shape=(text_dim,), name='text_input')
        
        # Feature branch
        feature_branch = Dense(64, activation='relu')(feature_input)
        feature_branch = Dropout(0.3)(feature_branch)
        feature_branch = Dense(32, activation='relu')(feature_branch)
        
        # Text branch
        text_branch = Dense(128, activation='relu')(text_input)
        text_branch = Dropout(0.3)(text_branch)
        text_branch = Dense(64, activation='relu')(text_branch)
        
        # Combine branches
        combined = Concatenate()([feature_branch, text_branch])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        
        # Output layer
        output = Dense(1, activation='linear')(combined)
        
        # Create model
        model = Model(inputs=[feature_input, text_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train_models(self, dataset_path=None):
        """Train TensorFlow models for resume scoring"""
        # If no dataset is provided, download a sample dataset
        if not dataset_path:
            logger.info("No dataset provided. Using synthetic data for training.")
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data(500)
        else:
            # Load the dataset
            try:
                data = pd.read_csv(dataset_path)
                logger.info(f"Loaded dataset from {dataset_path}")
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                data = self._generate_synthetic_data(500)
        
        # Preprocess text data
        if 'resume_text' in data.columns:
            data['processed_text'] = data['resume_text'].apply(self._preprocess_text)
        
        # Extract features
        features = []
        for _, row in data.iterrows():
            if 'resume_text' in data.columns and 'job_description' in data.columns:
                feature_dict = self._extract_features(row['resume_text'], row['job_description'])
                features.append(feature_dict)
        
        if features:
            features_df = pd.DataFrame(features)
        else:
            # Use only the text features if we couldn't extract structured features
            features_df = pd.DataFrame(index=data.index)
        
        # Create TF-IDF vectorizer for text
        if 'processed_text' in data.columns:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            text_features = self.vectorizer.fit_transform(data['processed_text'])
            text_features_array = text_features.toarray()
            
            # Save the vectorizer
            self._save_joblib_model(self.vectorizer, self.vectorizer_path)
        else:
            # Create dummy text features if no text is available
            text_features_array = np.zeros((len(data), 10))
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Save the scaler
        self._save_joblib_model(self.scaler, self.scaler_path)
        
        # Train ATS score model
        if 'ats_score' in data.columns:
            y_ats = data['ats_score'].values / 100.0  # Normalize to 0-1 range
            
            # Split data
            X_train_feat, X_test_feat, X_train_text, X_test_text, y_train, y_test = train_test_split(
                scaled_features, text_features_array, y_ats, test_size=0.2, random_state=42
            )
            
            # Create and train the model
            self.ats_model = self._create_tf_model(X_train_feat.shape[1], X_train_text.shape[1])
            
            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=self.ats_model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Train the model
            history = self.ats_model.fit(
                [X_train_feat, X_train_text], y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate the model
            y_pred = self.ats_model.predict([X_test_feat, X_test_text]).flatten() * 100.0  # Convert back to 0-100 range
            y_test_original = y_test * 100.0  # Convert back to 0-100 range
            mse = mean_squared_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            logger.info(f"ATS Score Model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Save training history and evaluation metrics
            self._save_training_results('ats_score', history.history, mse, r2)
            
            # Save the model
            self._save_tf_model(self.ats_model, self.ats_model_path)
        
        # Train Format score model
        if 'format_score' in data.columns:
            y_format = data['format_score'].values / 100.0  # Normalize to 0-1 range
            
            # Split data
            X_train_feat, X_test_feat, X_train_text, X_test_text, y_train, y_test = train_test_split(
                scaled_features, text_features_array, y_format, test_size=0.2, random_state=42
            )
            
            # Create and train the model
            self.format_model = self._create_tf_model(X_train_feat.shape[1], X_train_text.shape[1])
            
            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=self.format_model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Train the model
            history = self.format_model.fit(
                [X_train_feat, X_train_text], y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate the model
            y_pred = self.format_model.predict([X_test_feat, X_test_text]).flatten() * 100.0  # Convert back to 0-100 range
            y_test_original = y_test * 100.0  # Convert back to 0-100 range
            mse = mean_squared_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            logger.info(f"Format Score Model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Save training history and evaluation metrics
            self._save_training_results('format_score', history.history, mse, r2)
            
            # Save the model
            self._save_tf_model(self.format_model, self.format_model_path)
        
        # Train Section score model
        if 'section_score' in data.columns:
            y_section = data['section_score'].values / 100.0  # Normalize to 0-1 range
            
            # Split data
            X_train_feat, X_test_feat, X_train_text, X_test_text, y_train, y_test = train_test_split(
                scaled_features, text_features_array, y_section, test_size=0.2, random_state=42
            )
            
            # Create and train the model
            self.section_model = self._create_tf_model(X_train_feat.shape[1], X_train_text.shape[1])
            
            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=self.section_model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Train the model
            history = self.section_model.fit(
                [X_train_feat, X_train_text], y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate the model
            y_pred = self.section_model.predict([X_test_feat, X_test_text]).flatten() * 100.0  # Convert back to 0-100 range
            y_test_original = y_test * 100.0  # Convert back to 0-100 range
            mse = mean_squared_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            logger.info(f"Section Score Model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Save training history and evaluation metrics
            self._save_training_results('section_score', history.history, mse, r2)
            
            # Save the model
            self._save_tf_model(self.section_model, self.section_model_path)
        
        logger.info("TensorFlow model training completed")
    
    def _save_training_results(self, model_name, history, mse, r2):
        """Save training history and evaluation metrics"""
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'mse': mse,
            'r2': r2,
            'training_loss': history['loss'],
            'validation_loss': history['val_loss'],
            'training_mae': history['mae'],
            'validation_mae': history['val_mae']
        }
        
        # Save as CSV
        results_df = pd.DataFrame({
            'epoch': range(1, len(history['loss']) + 1),
            'training_loss': history['loss'],
            'validation_loss': history['val_loss'],
            'training_mae': history['mae'],
            'validation_mae': history['val_mae']
        })
        
        results_file = os.path.join(self.results_dir, f'{model_name}_training_history.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"Training history saved to {results_file}")
        
        # Save metrics summary
        metrics_file = os.path.join(self.results_dir, f'{model_name}_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")
            f.write(f"Final Training Loss: {history['loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
        
        logger.info(f"Metrics summary saved to {metrics_file}")
    
    def _generate_synthetic_data(self, n_samples=500):
        """Generate synthetic data for training"""
        logger.info(f"Generating {n_samples} synthetic resume samples for training")
        
        # Generate random resume texts of varying quality
        np.random.seed(42)
        data = []
        
        # Common resume sections and keywords
        sections = ['CONTACT INFORMATION', 'SUMMARY', 'EXPERIENCE', 'EDUCATION', 'SKILLS']
        skills = ['Python', 'Java', 'JavaScript', 'SQL', 'Machine Learning', 'Data Analysis', 'Project Management', 'Communication']
        
        for i in range(n_samples):
            # Determine quality level (0-4, with 4 being highest quality)
            quality = np.random.randint(0, 5)
            
            # Generate resume text based on quality
            resume_text = ""
            
            # Add sections based on quality
            included_sections = np.random.choice(sections, size=min(quality+1, len(sections)), replace=False)
            
            for section in included_sections:
                resume_text += section + "\n"
                
                # Add content based on section
                if section == 'CONTACT INFORMATION':
                    if quality >= 2:
                        resume_text += "Email: example@email.com\n"
                        resume_text += "Phone: (123) 456-7890\n"
                    if quality >= 3:
                        resume_text += "LinkedIn: linkedin.com/in/example\n"
                    if quality >= 4:
                        resume_text += "Portfolio: example.com\n"
                
                elif section == 'SUMMARY':
                    summary_length = quality * 20  # Higher quality = longer summary
                    resume_text += f"Professional with {quality+1} years of experience." + " " * summary_length + "\n"
                
                elif section == 'EXPERIENCE':
                    # Add 1-3 job experiences based on quality
                    for j in range(min(quality, 3)):
                        resume_text += f"Job Title {j+1}\n"
                        resume_text += f"Company {j+1} | 20{20-j} - Present\n"
                        
                        # Add bullet points based on quality
                        for k in range(quality):
                            resume_text += f"• Accomplished task {k+1}\n"
                
                elif section == 'EDUCATION':
                    resume_text += "Bachelor's Degree\n"
                    resume_text += "University Name | 2015 - 2019\n"
                    if quality >= 3:
                        resume_text += "GPA: 3.8/4.0\n"
                
                elif section == 'SKILLS':
                    # Add skills based on quality
                    selected_skills = np.random.choice(skills, size=min(quality+2, len(skills)), replace=False)
                    resume_text += ", ".join(selected_skills) + "\n"
                
                resume_text += "\n"
            
            # Calculate scores based on quality
            ats_score = min(100, max(0, 20 * quality + np.random.normal(0, 5)))
            format_score = min(100, max(0, 20 * quality + np.random.normal(0, 10)))
            section_score = min(100, max(0, 20 * quality + np.random.normal(0, 7)))
            
            # Generate a simple job description
            job_description = "Looking for a professional with skills in " + ", ".join(np.random.choice(skills, size=3, replace=False))
            
            data.append({
                'resume_text': resume_text,
                'job_description': job_description,
                'ats_score': ats_score,
                'format_score': format_score,
                'section_score': section_score
            })
        
        return pd.DataFrame(data)
    
    def predict_scores(self, resume_text, job_description=None):
        """Predict scores for a resume using TensorFlow models"""
        # Check if models are loaded
        if not all([self.ats_model, self.format_model, self.section_model, self.vectorizer, self.scaler]):
            logger.warning("Models not loaded. Training new models with synthetic data.")
            self.train_models()
        
        # Preprocess text
        processed_text = self._preprocess_text(resume_text)
        
        # Extract features
        features = self._extract_features(resume_text, job_description)
        features_df = pd.DataFrame([features])
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Add text features using the vectorizer
        text_features = self.vectorizer.transform([processed_text])
        text_features_array = text_features.toarray()
        
        # Make predictions
        ats_score = self.ats_model.predict([scaled_features, text_features_array]).flatten()[0] * 100.0
        format_score = self.format_model.predict([scaled_features, text_features_array]).flatten()[0] * 100.0
        section_score = self.section_model.predict([scaled_features, text_features_array]).flatten()[0] * 100.0
        
        # Ensure scores are within 0-100 range
        ats_score = max(0, min(100, ats_score))
        format_score = max(0, min(100, format_score))
        section_score = max(0, min(100, section_score))
        
        return {
            'ats_score': round(ats_score),
            'format_score': round(format_score),
            'section_score': round(section_score),
            'timestamp': datetime.now().isoformat()
        }


# Function to download dataset if needed
def download_tf_resume_dataset(output_path='tf_resume_dataset.csv'):
    """Download a resume dataset for TensorFlow training"""
    try:
        # For demonstration, we'll create a synthetic dataset
        model = ResumeTFModel()
        data = model._generate_synthetic_data(1000)
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to results directory
        file_path = os.path.join(results_dir, output_path)
        data.to_csv(file_path, index=False)
        logger.info(f"Synthetic dataset created at {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    # Create and train the model
    model = ResumeTFModel()
    
    # Download dataset if needed
    dataset_path = download_tf_resume_dataset()
    
    # Train the model
    model.train_models(dataset_path)
    
    # Test with a sample resume
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
    
    # Predict scores
    scores = model.predict_scores(sample_resume, sample_job)
    print("Predicted scores:")
    print(f"ATS Score: {scores['ats_score']}")
    print(f"Format Score: {scores['format_score']}")
    print(f"Section Score: {scores['section_score']}")
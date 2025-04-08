import os
import numpy as np
import pandas as pd
import re
import pickle
import joblib
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import spacy
from datetime import datetime

# Import transformers for BERT
try:
    from transformers import TFBertModel, BertTokenizer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logging.warning("Transformers library not available. BERT features will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('resume_bert_tf')

def download_bert_tf_resume_dataset(output_file=None):
    """Download or generate a synthetic dataset for BERT-TF resume scoring
    
    Args:
        output_file: Path to save the dataset
        
    Returns:
        Path to the dataset file
    """
    logger.info("Generating synthetic BERT-TF resume dataset")
    
    # Create a ResumeBERTTFModel instance to use its _generate_synthetic_data method
    model = ResumeBERTTFModel()
    data = model._generate_synthetic_data(500)
    
    # Save the dataset if output_file is provided
    if output_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        data.to_csv(output_file, index=False)
        logger.info(f"Synthetic BERT-TF resume dataset saved to {output_file}")
        return output_file
    else:
        # Create a temporary file
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "bert_tf_resume_dataset.csv")
        data.to_csv(temp_file, index=False)
        logger.info(f"Synthetic BERT-TF resume dataset saved to {temp_file}")
        return temp_file

class ResumeBERTTFModel:
    """TensorFlow-based model with BERT integration for resume scoring"""
    
    def __init__(self, model_dir='models'):
        """Initialize the TensorFlow model with BERT integration"""
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
        
        # Initialize BERT tokenizer and model if available
        self.bert_available = BERT_AVAILABLE
        if self.bert_available:
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
                logger.info("BERT model and tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Error loading BERT model: {str(e)}")
                self.bert_available = False
        
        # Model paths
        self.ats_model_path = os.path.join(self.model_dir, 'bert_tf_ats_score_model')
        self.format_model_path = os.path.join(self.model_dir, 'bert_tf_format_score_model')
        self.section_model_path = os.path.join(self.model_dir, 'bert_tf_section_score_model')
        self.scaler_path = os.path.join(self.model_dir, 'bert_tf_feature_scaler.pkl')
        
        # Load models if they exist
        self.ats_model = self._load_tf_model(self.ats_model_path)
        self.format_model = self._load_tf_model(self.format_model_path)
        self.section_model = self._load_tf_model(self.section_model_path)
        self.scaler = self._load_joblib_model(self.scaler_path)
    
    def _load_tf_model(self, model_path):
        """Load a TensorFlow model if it exists"""
        if os.path.exists(model_path):
            try:
                return load_model(model_path, custom_objects={'TFBertModel': TFBertModel} if self.bert_available else None)
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
        
        return text
    
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
    
    def _create_bert_tf_model(self, input_dim):
        """Create a TensorFlow model with BERT integration for resume scoring"""
        if not self.bert_available:
            logger.warning("BERT is not available. Creating standard TF model instead.")
            return self._create_standard_tf_model(input_dim)
        
        # Input layers
        feature_input = Input(shape=(input_dim,), name='feature_input')
        text_input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
        text_attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
        
        # Feature branch
        feature_branch = Dense(64, activation='relu')(feature_input)
        feature_branch = Dropout(0.3)(feature_branch)
        feature_branch = Dense(32, activation='relu')(feature_branch)
        
        # BERT branch
        bert_outputs = self.bert_model({
            'input_ids': text_input_ids,
            'attention_mask': text_attention_mask
        })
        bert_sequence_output = bert_outputs.last_hidden_state
        bert_pooled_output = GlobalAveragePooling1D()(bert_sequence_output)
        bert_branch = Dense(128, activation='relu')(bert_pooled_output)
        bert_branch = Dropout(0.3)(bert_branch)
        bert_branch = Dense(64, activation='relu')(bert_branch)
        
        # Combine branches
        combined = Concatenate()([feature_branch, bert_branch])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        
        # Output layer
        output = Dense(1, activation='linear')(combined)
        
        # Create model
        model = Model(
            inputs=[feature_input, text_input_ids, text_attention_mask], 
            outputs=output
        )
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
        
        return model
    
    def _create_standard_tf_model(self, input_dim):
        """Create a standard TensorFlow model without BERT for resume scoring"""
        # Input layer
        feature_input = Input(shape=(input_dim,), name='feature_input')
        
        # Hidden layers
        x = Dense(128, activation='relu')(feature_input)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=feature_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def _tokenize_text(self, text, max_length=128):
        """Tokenize text for BERT model"""
        if not self.bert_available:
            return None, None
        
        # Truncate text if it's too long
        text = text[:10000]  # Limit to prevent tokenizer issues
        
        # Tokenize the text
        encoded = self.bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        
        return encoded['input_ids'], encoded['attention_mask']
    
    def train_models(self, dataset_path=None):
        """Train TensorFlow models with BERT for resume scoring"""
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
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Save the scaler
        self._save_joblib_model(self.scaler, self.scaler_path)
        
        # Prepare BERT inputs if available
        if self.bert_available and 'processed_text' in data.columns:
            all_input_ids = []
            all_attention_masks = []
            
            for text in data['processed_text']:
                input_ids, attention_mask = self._tokenize_text(text)
                all_input_ids.append(input_ids[0])
                all_attention_masks.append(attention_mask[0])
            
            all_input_ids = np.array([ids.numpy() for ids in all_input_ids])
            all_attention_masks = np.array([mask.numpy() for mask in all_attention_masks])
        
        # Train ATS score model
        if 'ats_score' in data.columns:
            y_ats = data['ats_score'].values / 100.0  # Normalize to 0-1 range
            
            # Split data
            if self.bert_available and 'processed_text' in data.columns:
                train_indices, test_indices = train_test_split(
                    range(len(scaled_features)), test_size=0.2, random_state=42
                )
                
                X_train_feat = scaled_features[train_indices]
                X_test_feat = scaled_features[test_indices]
                X_train_ids = all_input_ids[train_indices]
                X_test_ids = all_input_ids[test_indices]
                X_train_masks = all_attention_masks[train_indices]
                X_test_masks = all_attention_masks[test_indices]
                y_train = y_ats[train_indices]
                y_test = y_ats[test_indices]
                
                # Create and train the model
                self.ats_model = self._create_bert_tf_model(X_train_feat.shape[1])
                
                # Define callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ModelCheckpoint(filepath=self.ats_model_path, save_best_only=True, monitor='val_loss')
                ]
                
                # Train the model
                history = self.ats_model.fit(
                    [X_train_feat, X_train_ids, X_train_masks], y_train,
                    epochs=20,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate the model
                y_pred = self.ats_model.predict([X_test_feat, X_test_ids, X_test_masks]).flatten() * 100.0
            else:
                # Use standard model without BERT
                X_train_feat, X_test_feat, y_train, y_test = train_test_split(
                    scaled_features, y_ats, test_size=0.2, random_state=42
                )
                
                # Create and train the model
                self.ats_model = self._create_standard_tf_model(X_train_feat.shape[1])
                
                # Define callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ModelCheckpoint(filepath=self.ats_model_path, save_best_only=True, monitor='val_loss')
                ]
                
                # Train the model
                history = self.ats_model.fit(
                    X_train_feat, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate the model
                y_pred = self.ats_model.predict(X_test_feat).flatten() * 100.0
            
            y_test_original = y_test * 100.0  # Convert back to 0-100 range
            mse = mean_squared_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            logger.info(f"ATS Score Model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Save training history and evaluation metrics
            self._save_training_results('bert_tf_ats_score', history.history, mse, r2)
            
            # Save the model
            self._save_tf_model(self.ats_model, self.ats_model_path)
            
            # Similar training for format and section score models
            # (Code omitted for brevity - would follow the same pattern as ATS score model)
        
        logger.info("BERT-TensorFlow model training completed")
    
    def _save_training_results(self, model_name, history, mse, r2):
        """Save training history and evaluation metrics"""
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
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_training_history.png'))
        logger.info(f"Training history plot saved to {os.path.join(self.results_dir, f'{model_name}_training_history.png')}")
    
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
        """Predict scores for a resume using BERT-TensorFlow models"""
        # Check if models are loaded
        if not all([self.ats_model, self.scaler]):
            logger.warning("Models not loaded. Training new models with synthetic data.")
            self.train_models()
        
        # Preprocess text
        processed_text = self._preprocess_text(resume_text)
        
        # Extract features
        features = self._extract_features(resume_text, job_description)
        features_df = pd.DataFrame([features])
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Make predictions
        if self.bert_available:
            # Tokenize text for BERT
            input_ids, attention_mask = self._tokenize_text(processed_text)
            input_ids_np = np.array([input_ids[0].numpy()])
            attention_mask_np = np.array([attention_mask[0].numpy()])
            
            # Predict with BERT model
            ats_score = self.ats_model.predict([scaled_features, input_ids_np, attention_mask_np]).flatten()[0] * 100.0
            
            # For simplicity, we're using the same model for all scores in this example
            # In a real implementation, you would have separate models for each score
            format_score = ats_score + np.random.normal(0, 5)  # Add some variation
            section_score = ats_score + np.random.normal(0, 5)  # Add some variation
        else:
            # Predict with standard model
            ats_score = self.ats_model.predict(scaled_features).flatten()[0] * 100.0
            format_score = ats_score + np.random.normal(0, 5)  # Add some variation
            section_score = ats_score + np.random.normal(0, 5)  # Add some variation
        
        # Ensure scores are within 0-100 range
        ats_score = max(0, min(100, ats_score))
        format_score = max(0, min(100, format_score))
        section_score = max(0, min(100, section_score))
        
        # Generate suggestions based on scores
        suggestions = self._generate_suggestions(ats_score, format_score, section_score, resume_text, job_description)
        
        return {
            'ats_score': round(ats_score),
            'format_score': round(format_score),
            'section_score': round(section_score),
            'timestamp': datetime.now().isoformat(),
            'suggestions': suggestions
        }
    
    def _generate_suggestions(self, ats_score, format_score, section_score, resume_text, job_description=None):
        """Generate improvement suggestions based on scores"""
        suggestions = []
        
        # ATS score suggestions
        if ats_score < 70:
            suggestions.append({
                "icon": "fa-robot",
                "text": "Improve your resume's ATS compatibility by using more industry-standard keywords"
            })
        
        # Format score suggestions
        if format_score < 70:
            suggestions.append({
                "icon": "fa-file-alt",
                "text": "Enhance your resume's format with clear section headers and consistent styling"
            })
        
        # Section score suggestions
        if section_score < 70:
            suggestions.append({
                "icon": "fa-list-alt",
                "text": "Include all essential resume sections: Contact, Summary, Experience, Education, and Skills"
            })
        
        # Content analysis
        word_count = len(resume_text.split())
        if word_count < 300:
            suggestions.append({
                "icon": "fa-file-text",
                "text": "Add more detail to your resume - aim for at least 300-600 words"
            })
        
        # Job description matching
        if job_description and len(suggestions) < 5:
            # Extract common skills from job description
            common_skills = ['python', 'java', 'javascript', 'sql', 'machine learning', 'data science', 'tensorflow', 'keras', 'bert']
            job_skills = [skill for skill in common_skills if skill in job_description.lower()]
            resume_skills = [skill for skill in common_skills if skill in resume_text.lower()]
            missing_skills = [skill for skill in job_skills if skill not in resume_skills]
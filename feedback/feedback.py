import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import time

class FeedbackManager:
    def __init__(self):
        self.db_path = "feedback/feedback.db"
        self.setup_database()

    def setup_database(self):
        """Create feedback table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rating INTEGER,
                usability_score INTEGER,
                feature_satisfaction INTEGER,
                missing_features TEXT,
                improvement_suggestions TEXT,
                user_experience TEXT,
                timestamp DATETIME
            )
        ''')
        conn.commit()
        conn.close()

    def save_feedback(self, feedback_data):
        """Save feedback to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO feedback (
                rating, usability_score, feature_satisfaction,
                missing_features, improvement_suggestions,
                user_experience, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_data['rating'],
            feedback_data['usability_score'],
            feedback_data['feature_satisfaction'],
            feedback_data['missing_features'],
            feedback_data['improvement_suggestions'],
            feedback_data['user_experience'],
            datetime.now()
        ))
        conn.commit()
        conn.close()

    def get_feedback_stats(self):
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()
        
        if df.empty:
            return {
                'avg_rating': 0,
                'avg_usability': 0,
                'avg_satisfaction': 0,
                'total_responses': 0
            }
        
        return {
            'avg_rating': df['rating'].mean(),
            'avg_usability': df['usability_score'].mean(),
            'avg_satisfaction': df['feature_satisfaction'].mean(),
            'total_responses': len(df)
        }

    def render_feedback_form(self):
        """Render the feedback form"""
        st.markdown("""
            <style>
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
            
            .feedback-container {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                padding: 30px;
                border-radius: 20px;
                margin: 20px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .feedback-header {
                color: #E0E0E0;
                font-size: 1.5em;
                font-weight: 600;
                margin-bottom: 25px;
                text-align: center;
                padding: 15px;
                background: linear-gradient(135deg, #4CAF50, #2196F3);
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
            }
            
            .feedback-section {
                margin: 20px 0;
                padding: 20px;
                border-radius: 15px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .feedback-section:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .feedback-label {
                color: #E0E0E0;
                font-size: 1.1em;
                font-weight: 500;
                margin-bottom: 10px;
            }
            
            .star-rating {
                font-size: 24px;
                color: #FFD700;
                cursor: pointer;
                transition: transform 0.2s ease;
            }
            
            .star-rating:hover {
                transform: scale(1.1);
            }
            
            .rating-container {
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 15px 0;
            }
            
            .submit-button {
                background: linear-gradient(135deg, #4CAF50, #2196F3);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
                width: 100%;
                margin-top: 20px;
            }
            
            .submit-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
            }
            
            .textarea-container {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
            }
            
            .textarea-container textarea {
                width: 100%;
                min-height: 100px;
                background: transparent;
                border: none;
                color: #E0E0E0;
                font-size: 1em;
                resize: vertical;
            }
            </style>
            """, unsafe_allow_html=True)

        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="feedback-header">📝 Share Your Feedback</h2>', unsafe_allow_html=True)

        # Overall Rating
        st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
        st.markdown('<label class="feedback-label">Overall Experience Rating</label>', unsafe_allow_html=True)
        rating = st.slider("Overall Rating", 1, 5, 5, help="Rate your overall experience with the app", label_visibility="collapsed")
        st.markdown(f'<div class="rating-container">{"⭐" * rating}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Usability Score
        st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
        st.markdown('<label class="feedback-label">How easy was it to use our app?</label>', unsafe_allow_html=True)
        usability_score = st.slider("Usability Score", 1, 5, 5, help="Rate the app's ease of use", label_visibility="collapsed")
        st.markdown(f'<div class="rating-container">{"⭐" * usability_score}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature Satisfaction
        st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
        st.markdown('<label class="feedback-label">How satisfied are you with our features?</label>', unsafe_allow_html=True)
        feature_satisfaction = st.slider("Feature Satisfaction", 1, 5, 5, help="Rate your satisfaction with the app's features", label_visibility="collapsed")
        st.markdown(f'<div class="rating-container">{"⭐" * feature_satisfaction}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Text Feedback
        st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
        st.markdown('<label class="feedback-label">What features would you like to see added?</label>', unsafe_allow_html=True)
        missing_features = st.text_area("Missing Features", placeholder="Share your feature requests...", label_visibility="collapsed")

        st.markdown('<label class="feedback-label">How can we improve?</label>', unsafe_allow_html=True)
        improvement_suggestions = st.text_area("Improvement Suggestions", placeholder="Your suggestions for improvement...", label_visibility="collapsed")

        st.markdown('<label class="feedback-label">Tell us about your experience</label>', unsafe_allow_html=True)
        user_experience = st.text_area("User Experience", placeholder="Share your experience with us...", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Submit Button
        if st.button("Submit Feedback", key="submit_feedback"):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing with animation
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("Processing feedback... 📝")
                    elif i < 60:
                        status_text.text("Analyzing responses... 🔍")
                    elif i < 90:
                        status_text.text("Saving to database... 💾")
                    else:
                        status_text.text("Finalizing... ✨")
                    time.sleep(0.01)

                # Save feedback
                feedback_data = {
                    'rating': rating,
                    'usability_score': usability_score,
                    'feature_satisfaction': feature_satisfaction,
                    'missing_features': missing_features,
                    'improvement_suggestions': improvement_suggestions,
                    'user_experience': user_experience
                }
                self.save_feedback(feedback_data)
                
                # Clear progress elements
                progress_bar.empty()
                status_text.empty()
                
                # Show success message with animation
                success_container = st.empty()
                success_container.markdown("""
                    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1)); border-radius: 10px;">
                        <h2 style="color: #4CAF50;">Thank You! 🎉</h2>
                        <p style="color: #E0E0E0;">Your feedback helps us improve Smart Resume AI</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show balloons animation
                st.balloons()
                
                # Keep success message visible
                time.sleep(2)
                
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")

    def render_feedback_stats(self):
        """Render feedback statistics"""
        stats = self.get_feedback_stats()
        
        st.markdown("""
            <div style="text-align: center; padding: 15px; background: linear-gradient(90deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1)); border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #E0E0E0;">Feedback Overview 📊</h3>
            </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(4)
        metrics = [
            {"label": "Total Responses", "value": f"{stats['total_responses']:,}", "delta": "↗"},
            {"label": "Avg Rating", "value": f"{stats['avg_rating']:.1f}/5.0", "delta": "⭐"},
            {"label": "Usability Score", "value": f"{stats['avg_usability']:.1f}/5.0", "delta": "🎯"},
            {"label": "Satisfaction", "value": f"{stats['avg_satisfaction']:.1f}/5.0", "delta": "😊"}
        ]
        
        for col, metric in zip(cols, metrics):
            col.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="color: #B0B0B0; font-size: 0.9em;">{metric['label']}</div>
                    <div style="font-size: 1.5em; color: #4CAF50; margin: 5px 0;">{metric['value']}</div>
                    <div style="color: #E0E0E0; font-size: 1.2em;">{metric['delta']}</div>
                </div>
            """, unsafe_allow_html=True)














# import streamlit as st
# import sqlite3
# from datetime import datetime
# import pandas as pd
# import time

# class FeedbackManager:
#     def __init__(self):
#         self.db_path = "feedback/feedback.db"
#         self.setup_database()

#     def setup_database(self):
#         """Create feedback table if it doesn't exist"""
#         conn = sqlite3.connect(self.db_path)
#         c = conn.cursor()
#         c.execute('''
#             CREATE TABLE IF NOT EXISTS feedback (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 rating INTEGER,
#                 usability_score INTEGER,
#                 feature_satisfaction INTEGER,
#                 missing_features TEXT,
#                 improvement_suggestions TEXT,
#                 user_experience TEXT,
#                 timestamp DATETIME
#             )
#         ''')
#         conn.commit()
#         conn.close()

#     def save_feedback(self, feedback_data):
#         """Save feedback to database"""
#         conn = sqlite3.connect(self.db_path)
#         c = conn.cursor()
#         c.execute('''
#             INSERT INTO feedback (
#                 rating, usability_score, feature_satisfaction,
#                 missing_features, improvement_suggestions,
#                 user_experience, timestamp
#             ) VALUES (?, ?, ?, ?, ?, ?, ?)
#         ''', (
#             feedback_data['rating'],
#             feedback_data['usability_score'],
#             feedback_data['feature_satisfaction'],
#             feedback_data['missing_features'],
#             feedback_data['improvement_suggestions'],
#             feedback_data['user_experience'],
#             datetime.now()
#         ))
#         conn.commit()
#         conn.close()

#     def get_feedback_stats(self):
#         """Get feedback statistics"""
#         conn = sqlite3.connect(self.db_path)
#         df = pd.read_sql_query("SELECT * FROM feedback", conn)
#         conn.close()
        
#         if df.empty:
#             return {
#                 'avg_rating': 0,
#                 'avg_usability': 0,
#                 'avg_satisfaction': 0,
#                 'total_responses': 0
#             }
        
#         return {
#             'avg_rating': df['rating'].mean(),
#             'avg_usability': df['usability_score'].mean(),
#             'avg_satisfaction': df['feature_satisfaction'].mean(),
#             'total_responses': len(df)
#         }

#     def render_feedback_form(self):
#         """Render the feedback form"""
#         st.markdown("""
#             <style>
#             @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
            
#             .feedback-container {
#                 background: rgba(255, 255, 255, 0.05);
#                 backdrop-filter: blur(10px);
#                 padding: 30px;
#                 border-radius: 20px;
#                 margin: 20px 0;
#                 border: 1px solid rgba(255, 255, 255, 0.1);
#                 box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
#             }
            
#             .feedback-header {
#                 color: #E0E0E0;
#                 font-size: 1.5em;
#                 font-weight: 600;
#                 margin-bottom: 25px;
#                 text-align: center;
#                 padding: 15px;
#                 background: linear-gradient(135deg, #4CAF50, #2196F3);
#                 border-radius: 12px;
#                 box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
#             }
            
#             .feedback-section {
#                 margin: 20px 0;
#                 padding: 20px;
#                 border-radius: 15px;
#                 background: rgba(255, 255, 255, 0.03);
#                 border: 1px solid rgba(255, 255, 255, 0.1);
#                 transition: transform 0.3s ease, box-shadow 0.3s ease;
#             }
            
#             .feedback-section:hover {
#                 transform: translateY(-5px);
#                 box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
#             }
            
#             .feedback-label {
#                 color: #E0E0E0;
#                 font-size: 1.1em;
#                 font-weight: 500;
#                 margin-bottom: 10px;
#             }
            
#             .star-rating {
#                 font-size: 24px;
#                 color: #FFD700;
#                 cursor: pointer;
#                 transition: transform 0.2s ease;
#             }
            
#             .star-rating:hover {
#                 transform: scale(1.1);
#             }
            
#             .rating-container {
#                 display: flex;
#                 align-items: center;
#                 gap: 10px;
#                 margin: 15px 0;
#             }
            
#             .submit-button {
#                 background: linear-gradient(135deg, #4CAF50, #2196F3);
#                 color: white;
#                 padding: 12px 25px;
#                 border: none;
#                 border-radius: 8px;
#                 font-weight: 600;
#                 cursor: pointer;
#                 transition: all 0.3s ease;
#                 text-transform: uppercase;
#                 letter-spacing: 1px;
#                 width: 100%;
#                 margin-top: 20px;
#             }
            
#             .submit-button:hover {
#                 transform: translateY(-2px);
#                 box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
#             }
            
#             .textarea-container {
#                 background: rgba(255, 255, 255, 0.03);
#                 border: 1px solid rgba(255, 255, 255, 0.1);
#                 border-radius: 8px;
#                 padding: 10px;
#                 margin-top: 10px;
#             }
            
#             .textarea-container textarea {
#                 width: 100%;
#                 min-height: 100px;
#                 background: transparent;
#                 border: none;
#                 color: #E0E0E0;
#                 font-size: 1em;
#                 resize: vertical;
#             }
#             </style>
#             """, unsafe_allow_html=True)

#         st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
#         st.markdown('<h2 class="feedback-header">📝 Share Your Feedback</h2>', unsafe_allow_html=True)

#         # Overall Rating
#         st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
#         st.markdown('<label class="feedback-label">Overall Experience Rating</label>', unsafe_allow_html=True)
#         rating = st.slider("", 1, 5, 5, help="Rate your overall experience with the app")
#         st.markdown(f'<div class="rating-container">{"⭐" * rating}</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#         # Usability Score
#         st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
#         st.markdown('<label class="feedback-label">How easy was it to use our app?</label>', unsafe_allow_html=True)
#         usability_score = st.slider("", 1, 5, 5, help="Rate the app's ease of use")
#         st.markdown(f'<div class="rating-container">{"⭐" * usability_score}</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#         # Feature Satisfaction
#         st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
#         st.markdown('<label class="feedback-label">How satisfied are you with our features?</label>', unsafe_allow_html=True)
#         feature_satisfaction = st.slider("", 1, 5, 5, help="Rate your satisfaction with the app's features")
#         st.markdown(f'<div class="rating-container">{"⭐" * feature_satisfaction}</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#         # Text Feedback
#         st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
#         st.markdown('<label class="feedback-label">What features would you like to see added?</label>', unsafe_allow_html=True)
#         missing_features = st.text_area("", placeholder="Share your feature requests...")

#         st.markdown('<label class="feedback-label">How can we improve?</label>', unsafe_allow_html=True)
#         improvement_suggestions = st.text_area("", placeholder="Your suggestions for improvement...")

#         st.markdown('<label class="feedback-label">Tell us about your experience</label>', unsafe_allow_html=True)
#         user_experience = st.text_area("", placeholder="Share your experience with us...")
#         st.markdown('</div>', unsafe_allow_html=True)

#         # Submit Button
#         if st.button("Submit Feedback", key="submit_feedback"):
#             # Create progress bar
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Simulate processing with animation
#             for i in range(100):
#                 progress_bar.progress(i + 1)
#                 if i < 30:
#                     status_text.text("Processing feedback... 📝")
#                 elif i < 60:
#                     status_text.text("Analyzing responses... 🔍")
#                 elif i < 90:
#                     status_text.text("Saving to database... 💾")
#                 else:
#                     status_text.text("Finalizing... ✨")
#                 time.sleep(0.01)

#             # Save feedback
#             feedback_data = {
#                 'rating': rating,
#                 'usability_score': usability_score,
#                 'feature_satisfaction': feature_satisfaction,
#                 'missing_features': missing_features,
#                 'improvement_suggestions': improvement_suggestions,
#                 'user_experience': user_experience
#             }
#             self.save_feedback(feedback_data)
            
#             # Clear progress elements
#             progress_bar.empty()
#             status_text.empty()
            
#             # Show success message with animation
#             success_container = st.empty()
#             success_container.markdown("""
#                 <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1)); border-radius: 10px;">
#                     <h2 style="color: #4CAF50;">Thank You! 🎉</h2>
#                     <p style="color: #E0E0E0;">Your feedback helps us improve Smart Resume AI</p>
#                 </div>
#             """, unsafe_allow_html=True)
            
#             st.balloons()
#             time.sleep(3)  # Keep success message visible for 3 seconds
#             success_container.empty()  # Clear success message

#     def render_feedback_stats(self):
#         """Render feedback statistics"""
#         stats = self.get_feedback_stats()
        
#         st.markdown("""
#             <div style="text-align: center; padding: 15px; background: linear-gradient(90deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1)); border-radius: 10px; margin-bottom: 20px;">
#                 <h3 style="color: #E0E0E0;">Feedback Overview 📊</h3>
#             </div>
#         """, unsafe_allow_html=True)
        
#         cols = st.columns(4)
#         metrics = [
#             {"label": "Total Responses", "value": f"{stats['total_responses']:,}", "delta": "↗"},
#             {"label": "Avg Rating", "value": f"{stats['avg_rating']:.1f}/5.0", "delta": "⭐"},
#             {"label": "Usability Score", "value": f"{stats['avg_usability']:.1f}/5.0", "delta": "🎯"},
#             {"label": "Satisfaction", "value": f"{stats['avg_satisfaction']:.1f}/5.0", "delta": "😊"}
#         ]
        
#         for col, metric in zip(cols, metrics):
#             col.markdown(f"""
#                 <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; text-align: center;">
#                     <div style="color: #B0B0B0; font-size: 0.9em;">{metric['label']}</div>
#                     <div style="font-size: 1.5em; color: #4CAF50; margin: 5px 0;">{metric['value']}</div>
#                     <div style="color: #E0E0E0; font-size: 1.2em;">{metric['delta']}</div>
#                 </div>
#             """, unsafe_allow_html=True)














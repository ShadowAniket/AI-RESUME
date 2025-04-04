import pandas as pd
from datetime import datetime

class ExcelManager:
    def __init__(self):
        self.excel_file = "resume_data.xlsx"
    
    def save_resume_data(self, user_id, job_role, content, analysis_data=None):
        try:
            # Try to read existing Excel file
            try:
                df = pd.read_excel(self.excel_file)
            except FileNotFoundError:
                df = pd.DataFrame()
            
            # Create new data entry
            new_data = {
                'user_id': user_id,
                'job_role': job_role,
                'content': content,
                'analysis_data': str(analysis_data) if analysis_data else None,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Append new data
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            
            # Save to Excel
            df.to_excel(self.excel_file, index=False)
            return True
        except Exception as e:
            print(f"Error saving to Excel: {str(e)}")
            return False
    
    def get_all_resumes(self):
        try:
            return pd.read_excel(self.excel_file)
        except FileNotFoundError:
            return pd.DataFrame()
    
    def get_user_resumes(self, user_id):
        df = self.get_all_resumes()
        return df[df['user_id'] == user_id]

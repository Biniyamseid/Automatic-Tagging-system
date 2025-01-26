import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.mlb = MultiLabelBinarizer()
    
    def load_data(self):
        """
        Load dataset from CSV or JSON
        Expected columns: 'text', 'tags'
        """
        try:
            df = pd.read_csv(self.data_path)
            # Assume tags are stored as list or string of comma-separated tags
            df['tags'] = df['tags'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
            return df
        except Exception as e:
            raise ValueError(f"Data loading error: {e}")
    
    def preprocess_text(self, text):
        """
        Basic text preprocessing
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def prepare_dataset(self, test_size=0.2, random_state=42):
        """
        Prepare training and testing datasets
        """
        df = self.load_data()
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Transform tags to binary matrix
        tags_binary = self.mlb.fit_transform(df['tags'])
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            tags_binary, 
            test_size=test_size, 
            random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'tag_classes': self.mlb.classes_
        }
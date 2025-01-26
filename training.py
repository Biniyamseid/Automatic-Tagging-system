import torch
from datasets import load_dataset
from src.data_processor import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.model import ModelTrainer
import numpy as np

def main():
    # Load Hugging Face dataset
    dataset = load_dataset("ag_news")
    
    # Prepare data
    texts = dataset['train']['text']
    labels = dataset['train']['label']
    
    # Convert labels to multi-label format
    unique_labels = list(set(labels))
    multi_hot_labels = np.eye(len(unique_labels))[labels]
    
    # Feature Extraction
    feature_extractor = FeatureExtractor(method='tfidf')
    X = feature_extractor.extract_features(texts)
    
    # Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, multi_hot_labels, test_size=0.2
    )
    
    # Model Training
    model_trainer = ModelTrainer(
        input_dim=X_train.shape[1], 
        num_classes=len(unique_labels)
    )
    
    model_trainer.train(X_train, y_train)
    
    # Evaluate
    y_pred = model_trainer.predict(X_test)
    metrics = model_trainer.evaluate(y_test, y_pred)
    
    print("Training Metrics:", metrics)
    
    # Save model
    import joblib
    joblib.dump(model_trainer.model, 'models/classifier.pkl')
    joblib.dump(feature_extractor.tfidf_vectorizer, 'models/vectorizer.pkl')

if __name__ == "__main__":
    main()
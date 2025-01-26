import torch
from datasets import load_dataset
from data_processor  import DataProcessor
from feature_extractor import FeatureExtractor
from model import ModelTrainer, MultiLabelClassifier
import numpy as np
import os
# Save model
import joblib
import torch.nn as nn

def main():

    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)
    # Load Hugging Face dataset
    dataset = load_dataset("ag_news")
    
    # Prepare data
    texts = dataset['train']['text']
    labels = dataset['train']['label']

       # Save some sample data for testing
    sample_data = {
        'texts': dataset['train']['text'][:5],  # Save first 5 examples
        'labels': dataset['train']['label'][:5]
    }
    joblib.dump(sample_data, 'models/sample_data.pkl')
    
    # Print samples for easy reference
    print("\nSample data saved for testing:")
    for text, label in zip(sample_data['texts'], sample_data['labels']):
        print(f"\nText: {text[:1000]}...")
        print(f"Label: {label}")
    
    # Feature Extraction
    feature_extractor = FeatureExtractor(method='tfidf')
    X = feature_extractor.extract_features(texts)
    
    # Instead of converting to one-hot format, keep labels as class indices
    labels = np.array(labels, dtype=np.int64)
    
    # Define the actual class names
    class_names = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }
    
    # Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create a validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Model Training
    model_trainer = ModelTrainer(
        input_dim=X_train.shape[1], 
        num_classes=len(class_names)
    )

    print(dataset['train'])
    
    # Train with validation
    model_trainer.train(X_train, y_train, X_val, y_val)
    
    # Final evaluation on test set
    test_metrics = model_trainer.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    
    joblib.dump(model_trainer.model, 'models/classifier.pkl')
    joblib.dump(feature_extractor.tfidf_vectorizer, 'models/vectorizer.pkl')

    # Save the class names
    joblib.dump(class_names, 'models/tag_classes.pkl')

if __name__ == "__main__":
    main()

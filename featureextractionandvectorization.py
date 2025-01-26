from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

class FeatureExtractor:
    def __init__(self, method='tfidf', pretrained_model='bert-base-uncased'):
        self.method = method
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
        # Transformer-based embedding
        if method == 'transformer':
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.transformer_model = AutoModel.from_pretrained(pretrained_model)
    
    def extract_features(self, texts):
        if self.method == 'tfidf':
            return self.tfidf_vectorizer.fit_transform(texts)
        
        elif self.method == 'transformer':
            # Transformer-based feature extraction
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=512, 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(embedding.numpy())
            
            return np.array(embeddings)
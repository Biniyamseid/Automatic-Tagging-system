import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.network(x))

class ModelTrainer:
    def __init__(self, input_dim, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiLabelClassifier(input_dim, num_classes).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            loss.backward()
            self.optimizer.step()
    
    def predict(self, X_test):
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return (predictions > 0.5).cpu().numpy()
    
    def evaluate(self, y_true, y_pred):
        return {
            'f1_score': f1_score(y_true, y_pred, average='micro'),
            'accuracy': accuracy_score(y_true, y_pred)
        }
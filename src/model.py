import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.toarray())
            outputs = self(X_tensor)
            print(f"Raw model outputs: {outputs.numpy()}")  # Debug print
            
            # Instead of thresholding, take the argmax to get single most likely class
            predictions = torch.zeros_like(outputs)
            max_indices = torch.argmax(outputs, dim=1)
            
            # Create one-hot encoded predictions
            for i, idx in enumerate(max_indices):
                predictions[i, idx] = 1
            
            return predictions.numpy()

class ModelTrainer:
    def __init__(self, input_dim, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiLabelClassifier(input_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20):
        # Training data
        X_tensor = torch.FloatTensor(X_train.toarray()).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Validation data
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.toarray()).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_accuracy = 100 * correct / total
            
            # Validation
            if X_val is not None:
                val_accuracy = self.evaluate(X_val, y_val)['accuracy'] * 100
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, '
                      f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, '
                      f'Train Acc: {train_accuracy:.2f}%')
    
    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            accuracy = (predicted == y_tensor).sum().item() / len(y)
            return {'accuracy': accuracy}
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def load_data(file_path):
    data = pd.read_csv(file_path)
    features = data.iloc[:, 200:2249]
    labels = data.iloc[:, -1]
    return features, labels

def create_dataloaders(features, labels, batch_size=32, test_size=0.2, val_size=0.1, n_components=50):
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features)
    
    features_tensor = torch.tensor(features_reduced, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.float32)
    
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features_tensor, labels_tensor, test_size=test_size + val_size, random_state=42)
    
    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features, temp_labels, test_size=test_size / (test_size + val_size), random_state=42)
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

class MorganMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MorganMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=10):
    train_losses = []
    val_losses = []
    val_true = []
    val_pred = []
    train_true = []
    train_pred = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_true.extend(batch_y.tolist())
            train_pred.extend(outputs.tolist())
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y.float())
                epoch_val_loss += loss.item()
                val_true.extend(batch_y.tolist())
                val_pred.extend(outputs.tolist())
        
        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses, train_true, train_pred, val_true, val_pred

def plot_auc(true_labels, pred_probs, title='Area Under Curve'):
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    auc_score = auc(recall, precision)
    
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC (area = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show()

if __name__ == "__main__":
    features, labels = load_data('data/train.csv')
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(features, labels)
    
    model = MorganMLP(input_dim=50, hidden_dim=256, output_dim=1)
    input_dim = next(iter(train_dataloader))[0].shape[1]
    model = MorganMLP(input_dim=input_dim, hidden_dim=256, output_dim=1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses, train_true, train_pred, val_true, val_pred = train_model(model, train_dataloader, val_dataloader, criterion, optimizer)
    
    plot_auc(train_true, train_pred, title='AUC Curve for Training Data')
    plot_auc(val_true, val_pred, title='AUC Curve for Validation Data')
    
    model.eval()
    test_loss = 0
    test_true = []
    test_pred = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            test_loss += loss.item()
            test_true.extend(batch_y.tolist())
            test_pred.extend(outputs.tolist())
    
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    plot_auc(test_true, test_pred, title='AUC Curve for Test Data')
    
    correct = 0
    total = 0
    for batch_X, batch_y in test_dataloader:
        outputs = model(batch_X).squeeze()
        predicted = (outputs > 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    def find_best_threshold(true_labels, pred_probs):
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
        f1_scores = 2 * recall * precision / (recall + precision)
        best_threshold = thresholds[f1_scores.argmax()]
        return best_threshold

    best_threshold = find_best_threshold(val_true, val_pred)
    print(f"Best Threshold: {best_threshold:.4f}")

    correct = 0
    total = 0
    for batch_X, batch_y in test_dataloader:
        outputs = model(batch_X).squeeze()
        predicted = (outputs > best_threshold).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy with Best Threshold: {accuracy:.4f}")

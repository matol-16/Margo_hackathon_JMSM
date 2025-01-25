import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv

import reduc_dim


# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.labels = []

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                co = row[:-1]
                self.data.append([float(i) for i in co])  # All columns except last as features
                self.labels.append(int(row[-1]))  # Last column as label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))
    
    def PCA(self,ncomponent):
        #réduit la dimensionnalité des données
        self.data = reduc_dim.pca(self.data,ncomponent)



# Define a simple neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    print("Training complete.")


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


# Main script
if __name__ == "__main__":
    # Load datasets
    train_dataset = CustomDataset('training3.csv')
    test_dataset = CustomDataset('test13.csv')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model parameters
    input_size = len(train_dataset[0][0])  # Number of features
    hidden_size = 64
    num_classes = len(set([label for _, label in train_dataset]))  # Number of classes

    # Initialize the model, loss function, and optimizer
    model = NeuralNet(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    # Evaluate the model
    evaluate_model(model, test_loader)
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully!")
import torch
import torch.nn as nn
import csv

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

def load_model(model_path, input_size, hidden_size, num_classes):
    model = NeuralNet(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_molecule_probabilities(model, molecule_list):
    with torch.no_grad():
        inputs = torch.tensor(molecule_list, dtype=torch.float32)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
        return probabilities.numpy()

def load_new_molecules(csv_file):
    molecules = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            co = [float(i) for i in row[:-1]]
            molecules.append(co)
    return molecules

def load_results(csv_file):
    molecules = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            co = [row[-1]]
            molecules.append(co)
    return molecules

if __name__ == "__main__":
    input_size = 199+2048
    hidden_size = 64
    num_classes = 2
    model_path = 'model.pth'

    # Load model
    model = load_model(model_path, input_size, hidden_size, num_classes)

    # Load new molecule data (ensure same preprocessing steps)
    new_molecules = load_new_molecules('testinggg.csv')
    results = load_results('testinggg.csv')
    probabilities = predict_molecule_probabilities(model, new_molecules)

    sum = 0
    tot = 0

    sum95 = 0
    tot95 = 0

    sum99 = 0
    tot99 = 0
    resultsb = []
    for element in results:
        resultsb.append(element[0])
    results = resultsb
    for idx, prob in enumerate(probabilities):
        if prob[0]>0.99:
            if float(results[idx])==0:
                sum99+=1
            tot99+=1
        if prob[1]>0.99:
            if float(results[idx])==1:
                sum99+=1
            tot99+=1
        ##################################
        if prob[0]>0.95:
            if float(results[idx])==0:
                sum95+=1
            tot95+=1
        if prob[1]>0.95:
            if float(results[idx])==1:
                sum95+=1
            tot95+=1
        ###################################
        if prob[0]>prob[1]:
            if float(results[idx])==0:
                sum+=1
            tot+=1
        if prob[1]>prob[0]:
            if float(results[idx])==1:
                sum+=1
            tot+=1

    print(tot, tot95, tot99)
    print(sum/tot, sum95/tot95, sum99/tot99)



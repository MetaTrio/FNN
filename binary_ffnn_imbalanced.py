import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FeedForwardNN, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(42, 1024)   #2mer 3mer
        self.fc2 = nn.Linear(1024, 512)  # Hidden layer with 512 neurons
        self.fc3 = nn.Linear(512, 256)  # Hidden layer with 256 neurons
        self.fc4 = nn.Linear(256, 128)   # Hidden layer with 128 neurons
        self.fc5 = nn.Linear(128, 1)     # Output layer with 1 neuron for binary classification
        
        # Activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with specified rate
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        # Forward pass with dropout layers after each hidden layer
        x = self.dropout(self.relu(self.fc1(x)))  # Input to 1st hidden layer
        x = self.dropout(self.relu(self.fc2(x)))  # 1st hidden to 2nd hidden layer
        x = self.dropout(self.relu(self.fc3(x)))  # 2nd hidden to 3rd hidden layer
        x = self.dropout(self.relu(self.fc4(x)))  # 3rd hidden to output layer
        x = self.fc5(x)  # Output layer without sigmoid

        return x
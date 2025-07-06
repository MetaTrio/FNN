import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    # def __init__(self, dropout_rate=0.5):
    #     super(FeedForwardNN, self).__init__()
        
    #     # Define the layers
    #     # self.fc1 = nn.Linear(32, 1024)   # Input layer with 32 features -- 3mer
    #     # self.fc1 = nn.Linear(136, 1024)   #4mer
    #     self.fc1 = nn.Linear(42, 1024)   #2mer 3mer
    #     self.fc2 = nn.Linear(1024, 512)  # Hidden layer with 512 neurons
    #     self.fc3 = nn.Linear(512, 256)  # Hidden layer with 256 neurons
    #     self.fc4 = nn.Linear(256, 128)   # Hidden layer with 128 neurons
    #     self.fc5 = nn.Linear(128, 1)     # Output layer with 1 neuron for binary classification
        
    #     # Activation function
    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with specified rate
    #     # self.sigmoid = nn.Sigmoid()
    
    # def forward(self, x):
    #     # Forward pass
    #     x = self.relu(self.fc1(x))  # Input to 1st hidden layer
    #     x = self.relu(self.fc2(x))  # 1st hidden to 2nd hidden layer
    #     x = self.relu(self.fc3(x))  # 2nd hidden to 3rd hidden layer
    #     x = self.relu(self.fc4(x)) 
    #     x = self.sigmoid(self.fc5(x))  # Sigmoid for binary classification
    #     # x = self.fc5(x)  # Output layer without sigmoid
        
    #     # Forward pass with dropout layers after each hidden layer
    #     # x = self.dropout(self.relu(self.fc1(x)))  # Input to 1st hidden layer
    #     # x = self.dropout(self.relu(self.fc2(x)))  # 1st hidden to 2nd hidden layer
    #     # x = self.dropout(self.relu(self.fc3(x)))  # 2nd hidden to 3rd hidden layer
    #     # x = self.dropout(self.relu(self.fc4(x)))  # 3rd hidden to output layer
    #     # x = self.fc5(x)  # Output layer without sigmoid   


    # for human_train_kraken_and_nano_len_7k_12k

    def __init__(self):
        super(FeedForwardNN, self).__init__()
        
        # Define the layers
        # self.fc1 = nn.Linear(32, 1024)   # Input layer with 3mer
        self.fc1 = nn.Linear(42, 1024)   # Input layer with 2mer, 3mer
        # self.fc1 = nn.Linear(2080, 1024)   #6mer
        # self.fc1 = nn.Linear(512, 1024)   #5mer
        # self.fc1 = nn.Linear(136, 1024)   # Input layer with 136 features -- 4mer
        self.fc2 = nn.Linear(1024, 256)  # Hidden layer with 256 neurons
        self.fc3 = nn.Linear(256, 128)   # Hidden layer with 128 neurons
        self.fc4 = nn.Linear(128, 1)     # Output layer with 1 neuron for binary classification
        
        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass
        x = self.relu(self.fc1(x))  # Input to 1st hidden layer
        x = self.relu(self.fc2(x))  # 1st hidden to 2nd hidden layer
        x = self.relu(self.fc3(x))  # 2nd hidden to 3rd hidden layer
        x = self.sigmoid(self.fc4(x))  # Sigmoid for binary classification
         
        return x

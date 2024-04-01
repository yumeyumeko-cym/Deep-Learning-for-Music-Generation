import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
hidden_size = 256
input_size = 18 # number of features in json file
num_layers = 2
output_unit = 18

# training parameters
n_epochs = 100   # number of epochs
batch_size = 64  # batch size



# load MusicDataset
input, target = generate_training_sequences_pytorch(SEQUENCE_LENGTH)

def data_splitter(X, v, batch_size, train_size):
    X_train, X_test, v_train, v_test = train_test_split(X, v, train_size=train_size, shuffle=True, random_state=1)
    
    # Assuming X and v are already tensors, just adjust dtype and device as needed.
    # If they're numpy arrays or lists, first conversion to tensor is needed, then no warning will be issued.
    X_train = X_train.clone().detach().to(dtype=torch.float32)
    v_train = v_train.clone().detach().to(dtype=torch.long)  # Ensure correct dtype for CrossEntropyLoss
    X_test = X_test.clone().detach().to(dtype=torch.float32)
    v_test = v_test.clone().detach().to(dtype=torch.long)

    return X_train, v_train, X_test, v_test


# Bi-LSTM model
class myBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(myBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.5)  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        # Since the LSTM is bidirectional, the output features are doubled
        self.batchnorm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes) 
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0)) 
        # Applying batch normalization to the output for the last time step from each direction
        lstm_out = self.batchnorm(lstm_out[:, -1, :])
        lstm_out = self.dropout(lstm_out) 
        out = self.fc(lstm_out) 
        return out

def training_loop(model, n_epochs, X_train, v_train, X_test, v_test):
    model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(X_train, v_train)
    test_dataset = TensorDataset(X_test, v_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }
    
    for epoch in range(n_epochs):
        model.train()
        for X_batch, v_batch in train_loader:
            X_batch, v_batch = X_batch.to(device), v_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, v_batch)
            loss.backward()
            optimizer.step()
        
        train_loss, train_accuracy = evaluate(model, train_loader, loss_fn, device)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    return history

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for X_batch, v_batch in data_loader:
            X_batch, v_batch = X_batch.to(device), v_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, v_batch)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += v_batch.size(0)
            correct_predictions += (predicted == v_batch).sum().item()
    
    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy






def main():
    input, target = generate_training_sequences_pytorch(SEQUENCE_LENGTH)
    X_train, v_train, X_test, v_test = data_splitter(input, target, batch_size, 0.8)
    
    bilstm_model = myBiLSTM(input_size, hidden_size, num_layers, output_unit)
    history = training_loop(bilstm_model, n_epochs, X_train, v_train, X_test, v_test)
    #train_risk, test_risk = training_loop(lstm_model, n_epochs, X_train, v_train, X_test, v_test)

    # save the model
    model_save_path = "trained_bi_lstm_model.pth"
    torch.save(bilstm_model.state_dict(), model_save_path)

    plt.figure(figsize=(12, 6))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['test_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    main()
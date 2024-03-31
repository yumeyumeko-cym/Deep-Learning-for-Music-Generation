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
n_epochs = 30   # number of epochs
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

def training_loop(model, n_epochs, batch_size, X_train, v_train, X_test, v_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_train, v_train)
    test_dataset = TensorDataset(X_test, v_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_risk, test_risk = [], []

    for epoch in tqdm(range(n_epochs), unit="epoch"):
        model.train()
        for X_batch, v_batch in train_loader:
            X_batch, v_batch = X_batch.to(device), v_batch.to(device)

            y_batch = model(X_batch)
            loss = loss_fn(y_batch, v_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        train_loss, test_loss = evaluate(model, train_loader, test_loader, loss_fn, device)
        train_risk.append(train_loss)
        test_risk.append(test_loss)

    return train_risk, test_risk

def evaluate(model, train_loader, test_loader, loss_fn, device):
    model.eval()
    train_loss, test_loss = 0.0, 0.0

    with torch.no_grad():
        for X_batch, v_batch in train_loader:
            X_batch, v_batch = X_batch.to(device), v_batch.to(device)
            y_train = model(X_batch)
            train_loss += loss_fn(y_train, v_batch).item()

        for X_batch, v_batch in test_loader:
            X_batch, v_batch = X_batch.to(device), v_batch.to(device)
            y_test = model(X_batch)
            test_loss += loss_fn(y_test, v_batch).item()

    train_loss /= len(train_loader.dataset)
    test_loss /= len(test_loader.dataset)

    return train_loss, test_loss






def main():
    # load MusicDataset
    input, target = generate_training_sequences_pytorch(SEQUENCE_LENGTH)
    #music_dataset = MusicDataset(input, target)
    X_train, v_train, X_test, v_test = data_splitter(input, target, batch_size, 0.8)
    
    lstm_model = myBiLSTM(input_size, hidden_size, num_layers, output_unit)
    train_risk, test_risk = training_loop(lstm_model, n_epochs, batch_size, X_train, v_train, X_test, v_test)

    # save the model
    model_save_path = "trained_bi_lstm_model.pth"
    torch.save(lstm_model.state_dict(), model_save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(train_risk, label='Training Loss')
    plt.plot(test_risk, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    main()
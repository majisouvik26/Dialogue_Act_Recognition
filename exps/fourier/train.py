# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fnet_model import FNetForDialogueActRecognition
from dataset import DialogueActDataset

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * input_ids.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * input_ids.size(0)
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

def main():
    d_model = 128              
    num_layers = 4            
    hidden_dim = 256           
    dropout = 0.1
    max_seq_length = 128
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DialogueActDataset(csv_file='dialogue_data.csv', max_seq_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.vocab)
    num_classes = len(dataset.labels)
    model = FNetForDialogueActRecognition(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_seq_length=max_seq_length
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss = train_model(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
if __name__ == "__main__":
    main()


# fnet_model.py
import torch
import torch.nn as nn
import torch.fft

class FourierTransform(nn.Module):
    def forward(self, x):
        x_fft = torch.fft.fft2(x, norm="ortho")
        return x_fft.real

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class FNetBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(FNetBlock, self).__init__()
        self.fourier = FourierTransform()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        fourier_out = self.fourier(x)
        x = self.layernorm1(x + self.dropout(fourier_out))
        
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_out))
        return x

class FNetForDialogueActRecognition(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_classes, hidden_dim, dropout=0.1, max_seq_length=128):
        super(FNetForDialogueActRecognition, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        self.layers = nn.ModuleList([
            FNetBlock(d_model, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        x = self.embedding(input_ids) 
        x = x + self.pos_embedding[:, :seq_length, :]
        for layer in self.layers:
            x = layer(x)
        
        x = x.mean(dim=1) 
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


import torch.nn as nn
import torch

class ContextAwareAttention(nn.Module):
    
    def __init__(self, hidden_size=1536, output_size=768, seq_len=128):
        super(ContextAwareAttention, self).__init__()
    
        self.fc_1 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        self.fc_3 = nn.Linear(in_features=hidden_size//2, out_features=output_size, bias=True)
        self.fc_2 = nn.Linear(in_features=output_size, out_features=128, bias=False)
        self.linear_projection = nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        
    
    def forward(self, hidden_states, h_forward):
        S = self.fc_2(torch.tanh(self.fc_1(hidden_states) + self.fc_3(h_forward.unsqueeze(1))))
        A = S.softmax(dim=-1)
        M = torch.matmul(A.permute(0, 2, 1), hidden_states)
        x = self.linear_projection(M)
        
        return x
        
        
        
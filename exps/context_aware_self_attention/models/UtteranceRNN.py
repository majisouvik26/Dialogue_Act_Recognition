
import  torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class UtteranceRNN(nn.Module):
    
    def __init__(self, model_name="roberta-base", hidden_size=768, bidirectional=True, num_layers=1):
        super(UtteranceRNN, self).__init__()
        
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        for param in self.base.parameters():
            param.requires_grad = False
        
        self.rnn = nn.RNN(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            batch_first=True
        )
    
    def forward(self, input_ids, attention_mask, seq_len):
        hidden_states, _ = self.base(input_ids, attention_mask)
        
        outputs,_ = self.rnn(hidden_states)
                
        return outputs
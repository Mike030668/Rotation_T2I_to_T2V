import torch
import torch.nn as nn
import torch.nn.functional as F
from  build_models.special_layers import CrossAttentionLayer, RotaryPositionalEmbedding
from  build_models.special_layers import ImprovedBlock

class Increment_spliter(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(Increment_spliter, self).__init__()
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.emb_dim = emb_dim
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)

        self.block_1 = ImprovedBlock(155, 256, 0.3).to(device)
        self.block_2 = ImprovedBlock(256, 128, 0.3).to(device)
        self.block_3 = ImprovedBlock(128, 64, 0.3).to(device)
        self.block_4 = ImprovedBlock(64, 32, 0.3).to(device)
        self.block_5 = ImprovedBlock(32, 16, 0.3).to(device)
        self.block_6 = ImprovedBlock(16, 8, 0.3).to(device)
        self.block_7 = ImprovedBlock(8, 4, 0.3).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)


    def forward(self, text_hidden_states, prior_embeds, rise):

        increment = self.lin_increment(rise).unsqueeze(1)
        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        prior_embeds = F.normalize(concat_data, p=2.0, dim=2)

        prior_trained = self.lin_start(prior_embeds)

        concat_data = torch.concat([text_hidden_states,
                                    prior_trained,
                                    cross_text_rise],
                                    axis=1)
        
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = 1)
        #print("concat_data", concat_data.shape)
        out = self.block_1(concat_data.permute(0, 2, 1))
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = self.block_6(out)
        out = self.block_7(out)
        # Use lin_final for the last step
        out = self.lin_final(out).permute(0, 2, 1)
        return out

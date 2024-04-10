import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len, device='cpu'):
        super(RotaryPositionalEmbedding, self).__init__()
        self.rotation_matrix = torch.zeros(d_model, d_model, device=device)
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(torch.tensor(i * j * 0.01, device=device))
        self.positional_embedding = torch.zeros(max_seq_len, d_model, device=device)
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(torch.tensor(i * j * 0.01, device=device))

    def forward(self, x):
        x += self.positional_embedding[:x.size(1), :]
        x = torch.matmul(x, self.rotation_matrix)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, emb_dim):
        super(CrossAttentionLayer, self).__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

    def forward(self, text_embeds, step_embeds):
        # Generating Q, K, V
        Q = self.query(text_embeds)  # Query from text embeddings
        # Key from combined step and direction embeddings
        K = self.key(step_embeds).permute(0, 2, 1)  
        V = self.value(step_embeds)  # Value also from merged embeddings

        # Calculate the attention weights and apply them to V
        attention_weights = F.softmax(torch.bmm(Q, K), dim=-1)
        attention_output = torch.bmm(attention_weights, V)
        # Return updated text embeddings
        return attention_output + text_embeds  


class ImprovedBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.3):
        super(ImprovedBlock, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        # Add a linear transformation for identity if the dimensions do not match
        if in_dim != out_dim:
            self.identity_mapping = nn.Linear(in_dim, out_dim)
        else:
            self.identity_mapping = None

    def forward(self, x):
        identity = x
        out = self.lin(x)
        out = self.act(out)
        out = self.norm(out)
        out = self.dropout(out)
        # Apply a linear transformation to identity if necessary
        if self.identity_mapping is not None:
            identity = self.identity_mapping(identity)
        out += identity  # Residual connection
        return out


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

        prior_trained = self.lin_start(prior_embeds)

        concat_data = torch.concat([text_hidden_states,
                                    prior_trained,
                                    cross_text_rise],
                                    axis=1)

        concat_data = F.normalize(concat_data, p=2.0, dim=1)
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

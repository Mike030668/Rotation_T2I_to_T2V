import torch
import torch.nn as nn

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

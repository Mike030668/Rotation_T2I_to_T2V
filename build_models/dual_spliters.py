import torch
import torch.nn as nn
import torch.nn.functional as F
from  build_models.special_layers import CrossAttentionLayer, RotaryPositionalEmbedding
from  build_models.special_layers import ImprovedBlock, ImprovedBlock_next



class DualBranchSpliter(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(DualBranchSpliter, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.emb_dim = emb_dim
        
        # Rise branch for handling rise-influenced data
        self.down_block_1 = nn.Sequential(
            ImprovedBlock(79, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_2 = nn.Sequential(
            ImprovedBlock(154, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_fin = nn.Sequential(
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 1, 0.3),
        ).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        # Positional encoding applied to text hidden states
        text_hidden_states = self.pos_encoder(text_hidden_states)

        # Base branch processing
        prior_trained = self.lin_start(prior_embeds)

        # Rise branch processing
        increment = self.lin_increment(rise).unsqueeze(1)

        concat_base = torch.concat([text_hidden_states,
                            prior_trained,
                            increment],
                            axis=1)

        cross_text_rise = self.cross_attention(text_hidden_states, increment)
        cross_text_prior = self.cross_attention(text_hidden_states, prior_trained)

        concat_cross = torch.concat([cross_text_rise,
                    cross_text_prior],
                    axis=1)


        base_output = self.down_block_1(concat_base.permute(0, 2, 1))

        cross_output = self.down_block_2(concat_cross.permute(0, 2, 1))


        concat_out = torch.concat([base_output,
                    cross_output],
                    axis=-1)

        out = self.down_block_fin(concat_out).permute(0, 2, 1)
        return out


class DualBranchSpliter_next(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(DualBranchSpliter_next, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block = nn.Sequential(
            ImprovedBlock_next(79, 128, nn.GELU),
            ImprovedBlock_next(128, 64, nn.GELU),
            ImprovedBlock_next(64, 32, nn.GELU),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_cross = nn.Sequential(
            ImprovedBlock_next(154, 256, nn.LeakyReLU),
            ImprovedBlock_next(256, 128, nn.LeakyReLU),
            ImprovedBlock_next(128, 64, nn.LeakyReLU),
            ImprovedBlock_next(64, 32, nn.LeakyReLU),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_fin = nn.Sequential(
            ImprovedBlock_next(64, 32, nn.SELU),
            ImprovedBlock_next(32, 16, nn.SELU),
            ImprovedBlock_next(16, 1, nn.SELU),
        ).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        # Positional encoding applied to text hidden states
        text_hidden_states = self.pos_encoder(text_hidden_states)

        # Base branch processing
        prior_trained = self.lin_start(prior_embeds)

        # Rise branch processing
        increment = self.lin_increment(rise).unsqueeze(1)


        concat_base = torch.concat([text_hidden_states,
                            prior_trained,
                            increment],
                            axis=1)


        cross_text_rise = self.cross_attention(text_hidden_states, increment)
        cross_text_prior = self.cross_attention(text_hidden_states, prior_trained)

        concat_cross = torch.concat([cross_text_rise,
                    cross_text_prior],
                    axis=1)

        base_output = self.down_block(concat_base.permute(0, 2, 1))
        cross_output = self.down_block_cross(concat_cross.permute(0, 2, 1))


        concat_out = torch.concat([base_output,
                    cross_output],
                    axis=-1)

        out = self.down_block_fin(concat_out).permute(0, 2, 1)
        return out
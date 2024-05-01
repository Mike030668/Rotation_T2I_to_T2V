import torch
import torch.nn as nn
import torch.nn.functional as F
from  build_models.special_layers import CrossAttentionLayer, RotaryPositionalEmbedding
from  build_models.special_layers import ImprovedBlock, ImprovedBlock_next


class Increment_spliter_0(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(Increment_spliter_0, self).__init__()
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


class Increment_spliter_1(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(Increment_spliter_1, self).__init__()
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
        increment =  nn.LeakyReLU()(increment)
        
        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        # nomolise espessialy for regress
        prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
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


class Increment_spliter_2(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(Increment_spliter_2, self).__init__()
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.emb_dim = emb_dim
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)

        self.block_1 = ImprovedBlock(232, 256, 0.3).to(device)
        self.block_2 = ImprovedBlock(256, 128, 0.3).to(device)
        self.block_3 = ImprovedBlock(128, 64, 0.3).to(device)
        self.block_4 = ImprovedBlock(64, 32, 0.3).to(device)
        self.block_5 = ImprovedBlock(32, 16, 0.3).to(device)
        self.block_6 = ImprovedBlock(16, 8, 0.3).to(device)
        self.block_7 = ImprovedBlock(8, 4, 0.3).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)


    def forward(self, text_hidden_states, prior_embeds, rise):

        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)
        
        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        # nomolise espessialy for regress
        prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        #prior_trained = self.lin_start(prior_embeds)
        cross_text_prior = self.cross_attention(text_hidden_states, prior_embeds)


        concat_data = torch.concat([text_hidden_states,
                                    prior_embeds,
                                    cross_text_prior,
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

        # nomolise espessialy for regress
        prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        # Base branch processing
        prior_trained = self.lin_start(prior_embeds)

        # Rise branch processing
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

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

class DualBranchSpliter_1(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(DualBranchSpliter_1, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.emb_dim = emb_dim
        
        # Rise branch for handling rise-influenced data
        self.down_block_1 = nn.Sequential(
            ImprovedBlock(156, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_2 = nn.Sequential(
            ImprovedBlock(77, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_fin = nn.Sequential(
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 4, 0.3),
        ).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):

        # Rise branch processing
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

        # Positional encoding applied to text hidden states
        text_hidden_states = self.pos_encoder(text_hidden_states)
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        # nomolise espessialy for regress
        prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        # Base branch processing
        prior_trained = self.lin_start(prior_embeds)
        cross_prior_rise = self.cross_attention(prior_trained, increment)

        concat_base = torch.concat([text_hidden_states,
                                    prior_trained,
                                    cross_prior_rise,
                                    cross_text_rise],
                                    axis=1)
        concat_base = torch.nn.functional.normalize(concat_base, p=2.0, dim = -1)
        base_output = self.down_block_1(concat_base.permute(0, 2, 1))

        cross_text_prior = self.cross_attention(text_hidden_states, prior_trained)

        cross_emb_output = self.down_block_2(cross_text_prior.permute(0, 2, 1))


        concat_out = torch.concat([base_output,
                                    cross_emb_output],
                                    axis=-1)
        concat_out = torch.nn.functional.normalize(concat_out, p=2.0, dim = -1)

        out = self.down_block_fin(concat_out)
        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)
        
        return out






class DualBranchSpliter_next_1(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(DualBranchSpliter_next_1, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.emb_dim = emb_dim
        # Rise branch for handling rise-influenced data
        self.down_block = nn.Sequential(
            ImprovedBlock_next(79, 128, nn.GELU),
            ImprovedBlock_next(128, 64, nn.GELU),
            ImprovedBlock_next(64, 32, nn.GELU),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_cross = nn.Sequential(
            ImprovedBlock_next(154, 256, nn.SELU),
            ImprovedBlock_next(256, 128, nn.SELU),
            ImprovedBlock_next(128, 64, nn.SELU),
            ImprovedBlock_next(64, 32, nn.SELU),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_fin = nn.Sequential(
            ImprovedBlock_next(64, 32, nn.GELU),
            ImprovedBlock_next(32, 16, nn.GELU),
            ImprovedBlock_next(16, 1, nn.GELU),
        ).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        # Positional encoding applied to text hidden states
        text_hidden_states = self.pos_encoder(text_hidden_states)

        # nomolise espessialy for regress
        prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        # Base branch processing
        prior_trained = self.lin_start(prior_embeds)

        # Rise branch processing
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

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


class DualBranchSpliter_next_2(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device='cpu'):
        super(DualBranchSpliter_next_2, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.emb_dim = emb_dim
        # Rise branch for handling rise-influenced data
        self.down_block = nn.Sequential(
            ImprovedBlock_next(79, 128, nn.GELU),
            ImprovedBlock_next(128, 64, nn.GELU),
            ImprovedBlock_next(64, 32, nn.GELU),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_cross = nn.Sequential(
            ImprovedBlock_next(154, 256, nn.SELU),
            ImprovedBlock_next(256, 128, nn.SELU),
            ImprovedBlock_next(128, 64, nn.SELU),
            ImprovedBlock_next(64, 32, nn.SELU),
        ).to(device)

        # Rise branch for handling rise-influenced data
        self.down_block_fin = nn.Sequential(
            ImprovedBlock_next(64, 32, nn.GELU),
            ImprovedBlock_next(32, 16, nn.GELU),
            ImprovedBlock_next(16, 1, nn.GELU),
        ).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        # Positional encoding applied to text hidden states
        text_hidden_states = self.pos_encoder(text_hidden_states)

        # Base branch processing
        prior_trained = self.lin_start(prior_embeds)
        prior_trained = nn.Tanh()(prior_trained)

        # Rise branch processing
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

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
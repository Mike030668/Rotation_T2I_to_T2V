import torch
import torch.nn as nn
import torch.nn.functional as F
from  build_models.special_layers import CrossAttentionLayer, RotaryPositionalEmbedding, ConsistentSelfAttention
from  build_models.special_layers import ConsistentSelfAttentionBase, ConsistentSelfAttentionTile
from  build_models.special_layers import ImprovedBlock, ImprovedBlock_next
from  utils.constats import EMB_DIM, MAX_SEQ_LEN_K22


class Increment_spliter_0(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
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
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
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
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
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

class Increment_spliter_3(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(Increment_spliter_3, self).__init__()
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.emb_dim = emb_dim
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        # blocks with trained dropout and sckit connections
        self.down_block = nn.Sequential(
            ImprovedBlock(157, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 8, 0.3),
            ImprovedBlock(8, 4, 0.3)
        ).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        # increment block
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        # normalise espessialy for regress
        #prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        prior_trained = self.lin_start(prior_embeds)
        cross_prior_rise = self.cross_attention(prior_trained, increment)

        cross_text_prior = self.cross_attention(prior_trained, text_hidden_states)
        cross_text_prior = torch.argmax(cross_text_prior, dim=1).unsqueeze(1)
 
        # concat block
        concat_data = torch.concat([text_hidden_states,
                                    prior_embeds,
                                    cross_prior_rise,
                                    cross_text_rise,
                                    cross_text_prior
                                    ],
                                    axis=1)
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = -1)

        # encode_block with trained dropout and sckit connections
        out = self.down_block(concat_data.permute(0, 2, 1))

        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)
        return out



class Increment_spliter_4(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(Increment_spliter_4, self).__init__()
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.emb_dim = emb_dim
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        # blocks with trained dropout and sckit connections
        self.down_block = nn.Sequential(
            ImprovedBlock(158, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 8, 0.3),
            ImprovedBlock(8, 4, 0.3)
        ).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        # increment block
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        # normalise espessialy for regress
        #prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        prior_trained = self.lin_start(prior_embeds)
        cross_prior_rise = self.cross_attention(prior_trained, increment)

        cross_text_prior = self.cross_attention(prior_trained, text_hidden_states)
        cross_text_prior = torch.argmax(cross_text_prior, dim=1).unsqueeze(1)
 
        # concat block
        concat_data = torch.concat([text_hidden_states,
                                    prior_embeds,
                                    increment,
                                    cross_prior_rise,
                                    cross_text_rise,
                                    cross_text_prior
                                    ],
                                    axis=1)
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = -1)

        # encode_block with trained dropout and sckit connections
        out = self.down_block(concat_data.permute(0, 2, 1))

        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)
        return out



class Increment_spliter_5(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(Increment_spliter_5, self).__init__()
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.emb_dim = emb_dim
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        # blocks with trained dropout and sckit connections
        self.down_block = nn.Sequential(
            ImprovedBlock(234, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 8, 0.3),
            ImprovedBlock(8, 4, 0.3)
        ).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):

        # increment block
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        # normalise espessialy for regress
        prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        prior_trained = self.lin_start(prior_embeds)
        cross_prior_rise = self.cross_attention(prior_trained, increment)

        cross_due_text_prior = self.cross_attention(cross_text_rise, cross_prior_rise)

        # concat block
        concat_data = torch.concat([text_hidden_states,
                                    prior_embeds,
                                    increment,
                                    cross_prior_rise,
                                    cross_text_rise,
                                    cross_due_text_prior
                                    ],
                                    axis=1)
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = -1)

        # encode_block with trained dropout and sckit connections
        out = self.down_block(concat_data.permute(0, 2, 1))

        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)
        return out



class Increment_spliter_5_1(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(Increment_spliter_5_1, self).__init__()
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.emb_dim = emb_dim
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        # blocks with trained dropout and sckit connections
        self.down_block = nn.Sequential(
            ImprovedBlock(234, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 8, 0.3),
            ImprovedBlock(8, 4, 0.3)
        ).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):

        # increment block
        increment = self.lin_increment(rise).unsqueeze(1)
        increment =  nn.LeakyReLU()(increment)

        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        # normalise espessialy for regress
        #prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        prior_trained = self.lin_start(prior_embeds)
        cross_prior_rise = self.cross_attention(prior_trained, increment)

        cross_due_text_prior = self.cross_attention(cross_text_rise, cross_prior_rise)

        # concat block
        concat_data = torch.concat([text_hidden_states,
                                    prior_trained, # prior_embeds
                                    increment,
                                    cross_prior_rise,
                                    cross_text_rise,
                                    cross_due_text_prior
                                    ],
                                    axis=1)
        
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = -1)

        # encode_block with trained dropout and sckit connections
        out = self.down_block(concat_data.permute(0, 2, 1))

        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)
        return out




class Increment_spliter_6(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(Increment_spliter_6, self).__init__()
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.emb_dim = emb_dim
        self.increment_block = ImprovedBlock(1, emb_dim, 0.3).to(device)
        self.prior_block = ImprovedBlock(emb_dim, emb_dim, 0.3).to(device)

        # blocks with trained dropout and sckit connections
        self.down_block = nn.Sequential(
            ImprovedBlock(234, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 8, 0.3),
            ImprovedBlock(8, 4, 0.3)
        ).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):

        # increment block
        increment = self.increment_block(rise).unsqueeze(1)
        # Use RotaryPositionalEmbedding
        text_hidden_states = self.pos_encoder(text_hidden_states)
        # Apply CrossAttentionLayer text_hidden_states and increment
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        #rise_m = rise.unsqueeze(1).repeat(1, 1, self.emb_dim)
        # multiply on rise_m
        #prior_embeds_m = prior_embeds*rise_m
        #prior_trained = self.prior_block(prior_embeds_m)
        # normalise espessialy for regress
        #prior_embeds =  torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
        prior_trained = self.prior_block(prior_embeds)


        cross_prior_rise = self.cross_attention(prior_trained, increment)

        cross_due_text_prior = self.cross_attention(cross_text_rise, cross_prior_rise)

        # concat block
        concat_data = torch.concat([text_hidden_states,
                                    prior_trained,
                                    increment,
                                    cross_prior_rise,
                                    cross_text_rise,
                                    cross_due_text_prior
                                    ],
                                    axis=1)
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = -1)

        # encode_block with trained dropout and sckit connections
        out = self.down_block(concat_data.permute(0, 2, 1))

        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)
        return out


class DualBranchSpliter(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
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
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(DualBranchSpliter_1, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.emb_dim = emb_dim
        
        # Rise branch for handling rise-influenced data
        self.down_block_1 = nn.Sequential(
            ImprovedBlock(157, 128, 0.3),
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

        concat_base = torch.concat([increment,
                                    text_hidden_states,
                                    prior_trained,
                                    cross_prior_rise,
                                    cross_text_rise],
                                    axis=1)
        
        concat_base = torch.nn.functional.normalize(concat_base, p=2.0, dim = -1)
        base_output = self.down_block_1(concat_base.permute(0, 2, 1))

        cross_text_prior = self.cross_attention(text_hidden_states, prior_trained)
        cross_text_prior = torch.nn.functional.normalize(cross_text_prior, p=2.0, dim = -1)
        cross_emb_output = self.down_block_2(cross_text_prior.permute(0, 2, 1))


        concat_out = torch.concat([base_output,
                                    cross_emb_output],
                                    axis=-1)
        concat_out = torch.nn.functional.normalize(concat_out, p=2.0, dim = -1)

        out = self.down_block_fin(concat_out)
        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)

        return out



class DualBranchSpliter_up(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(DualBranchSpliter_up, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_final = nn.Linear(4, 1).to(device)
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
            ImprovedBlock(16, 4, 0.3),
        ).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        # Positional encoding applied to text hidden states
        text_hidden_states = self.pos_encoder(text_hidden_states)

        prior_embeds = torch.nn.functional.normalize(prior_embeds, p=2.0, dim = -1)
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

        out = self.down_block_fin(concat_out)
        # next predicted prior_embeds
        out = self.lin_final(out).permute(0, 2, 1)
        return out



class DualBranchSpliter_next_1(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
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
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
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
    

#################### SELF ATTENTION #######################################################


class IncrementSpliterSA(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, max_seq_len=MAX_SEQ_LEN_K22, device='cpu'):
        super(IncrementSpliterSA, self).__init__()
        self.emb_dim = emb_dim
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.csa = ConsistentSelfAttentionBase(emb_dim).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)

        self.block_1 = ImprovedBlock(235, 256, 0.3).to(device)
        self.block_2 = ImprovedBlock(256, 128, 0.3).to(device)
        self.block_3 = ImprovedBlock(128, 64, 0.3).to(device)
        self.block_4 = ImprovedBlock(64, 32, 0.3).to(device)
        self.block_5 = ImprovedBlock(32, 16, 0.3).to(device)
        self.block_6 = ImprovedBlock(16, 8, 0.3).to(device)
        self.block_7 = ImprovedBlock(8, 4, 0.3).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        increment = self.lin_increment(rise).unsqueeze(1)
        increment = nn.LeakyReLU()(increment)

        text_hidden_states = self.pos_encoder(text_hidden_states)
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        prior_embeds = torch.nn.functional.normalize(prior_embeds, p=2.0, dim=-1)
        cross_text_prior = self.cross_attention(prior_embeds, increment)

        concat_base = torch.concat([text_hidden_states, prior_embeds], axis=1)
        self_attn = self.csa(concat_base)

        concat_data = torch.concat([increment,
                                    text_hidden_states,
                                    prior_embeds,
                                    cross_text_prior,
                                    cross_text_rise,
                                    self_attn
                                    ], axis=1)
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim=-1)

        out = self.block_1(concat_data.permute(0, 2, 1))
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = self.block_6(out)
        out = self.block_7(out)

        out = self.lin_final(out).permute(0, 2, 1)
        return out

class IncrementSpliterSAI(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, max_seq_len=MAX_SEQ_LEN_K22, device='cpu'):
        super(IncrementSpliterSAI, self).__init__()
        self.emb_dim = emb_dim
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.csa = ConsistentSelfAttentionBase(emb_dim).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)

        self.block_1 = ImprovedBlock(236, 256, 0.3).to(device)
        self.block_2 = ImprovedBlock(256, 128, 0.3).to(device)
        self.block_3 = ImprovedBlock(128, 64, 0.3).to(device)
        self.block_4 = ImprovedBlock(64, 32, 0.3).to(device)
        self.block_5 = ImprovedBlock(32, 16, 0.3).to(device)
        self.block_6 = ImprovedBlock(16, 8, 0.3).to(device)
        self.block_7 = ImprovedBlock(8, 4, 0.3).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        increment = self.lin_increment(rise).unsqueeze(1)
        increment = nn.LeakyReLU()(increment)

        text_hidden_states = self.pos_encoder(text_hidden_states)
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        prior_embeds = torch.nn.functional.normalize(prior_embeds, p=2.0, dim=-1)
        cross_text_prior = self.cross_attention(prior_embeds, increment)

        concat_base = torch.concat([text_hidden_states, prior_embeds, increment], axis=1)
        self_attn = self.csa(concat_base)

        concat_data = torch.concat([increment,
                                    text_hidden_states,
                                    prior_embeds,
                                    cross_text_prior,
                                    cross_text_rise,
                                    self_attn
                                    ], axis=1)
        
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim=-1)

        out = self.block_1(concat_data.permute(0, 2, 1))
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = self.block_6(out)
        out = self.block_7(out)

        out = self.lin_final(out).permute(0, 2, 1)
        return out


class IncrementSpliterSAT(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(IncrementSpliterSAT, self).__init__()
        self.emb_dim = emb_dim
        self.consistent_self_attention = ConsistentSelfAttentionTile(emb_dim, sampling_rate=0.5, tile_size=5).to(device)
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.emb_dim = emb_dim
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)

        self.block_1 = ImprovedBlock(235, 256, 0.3).to(device)
        self.block_2 = ImprovedBlock(256, 128, 0.3).to(device)
        self.block_3 = ImprovedBlock(128, 64, 0.3).to(device)
        self.block_4 = ImprovedBlock(64, 32, 0.3).to(device)
        self.block_5 = ImprovedBlock(32, 16, 0.3).to(device)
        self.block_6 = ImprovedBlock(16, 8, 0.3).to(device)
        self.block_7 = ImprovedBlock(8, 4, 0.3).to(device)

        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        increment = self.lin_increment(rise).unsqueeze(1)
        increment = nn.LeakyReLU()(increment)

        text_hidden_states = self.pos_encoder(text_hidden_states)
        cross_text_rise = self.cross_attention(text_hidden_states, increment)

        prior_embeds = torch.nn.functional.normalize(prior_embeds, p=2.0, dim=-1)
        cross_text_prior = self.cross_attention(prior_embeds, increment)

        concat_base = torch.concat([text_hidden_states, prior_embeds], axis=1)
        self_attn = self.consistent_self_attention(concat_base)

        concat_data = torch.concat([increment,
                                    text_hidden_states,
                                    prior_embeds,
                                    cross_text_prior,
                                    cross_text_rise,
                                    self_attn
                                    ], axis=1)
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim=-1)

        out = self.block_1(concat_data.permute(0, 2, 1))
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = self.block_6(out)
        out = self.block_7(out)

        out = self.lin_final(out).permute(0, 2, 1)
        return out



class DualBranchSpliterSA(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(DualBranchSpliterSA, self).__init__()

        self.emb_dim = emb_dim
        self.device = device

        self.csa = ConsistentSelfAttentionBase(emb_dim).to(device)
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)

        
        # Rise branch for handling rise-influenced data
        self.down_block_1 = nn.Sequential(
            ImprovedBlock(158, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
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
        increment = nn.LeakyReLU()(increment)

        concat_base = torch.concat([text_hidden_states,
                                    prior_trained,
                                    increment],
                                    axis=1)
        
        self_attn = self.csa(concat_base)
        concat_base = torch.concat([concat_base,
                                    self_attn],
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



class Increment_spliter_next_SA(nn.Module):
    def __init__(self, emb_dim = EMB_DIM, max_seq_len = MAX_SEQ_LEN_K22, device='cpu'):
        super(Increment_spliter_next_SA, self).__init__()
        self.emb_dim = emb_dim
        self.device = device

        self.csa = ConsistentSelfAttentionBase(emb_dim).to(device)
        # add CrossAttentionLayer
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        # add RotaryPositionalEmbedding
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)

        # others
        self.lin_increment= nn.Linear(1, emb_dim).to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.down_block = nn.Sequential(
            ImprovedBlock_next(235, 256, nn.GELU),
            ImprovedBlock_next(256, 128, nn.GELU),
            ImprovedBlock_next(128, 64, nn.GELU),
            ImprovedBlock_next(64, 32, nn.GELU),
            ImprovedBlock_next(32, 16, nn.GELU),
            ImprovedBlock_next(16, 8, nn.GELU),
            ImprovedBlock_next(8, 4, nn.GELU)
        ).to(device)

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
        cross_text_prior = self.cross_attention(prior_trained, increment)

        concat_base = torch.concat([text_hidden_states,
                                    prior_embeds,
                                    increment], axis=1)
        
        self_attn = self.csa(concat_base)

        concat_data = torch.concat([text_hidden_states,
                                    prior_embeds,
                                    cross_text_prior,
                                    cross_text_rise,
                                    self_attn
                                    ], axis=1)


        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim = -1)
        out = self.down_block(concat_data.permute(0, 2, 1))

        # Use lin_final for the last step
        out = self.lin_final(out).permute(0, 2, 1)
        return out

######################################################################################################

class Spliter_next_CSA(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, max_seq_len=MAX_SEQ_LEN_K22, device='cpu'):
        super(Spliter_next_CSA, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.consistent_self_attn = ConsistentSelfAttention(emb_dim).to(device)
        self.emb_dim = emb_dim
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.down_block = nn.Sequential(
            ImprovedBlock_next(156, 256, nn.GELU),
            ImprovedBlock_next(256, 128, nn.GELU),
            ImprovedBlock_next(128, 64, nn.GELU),
            ImprovedBlock_next(64, 32, nn.GELU),
            ImprovedBlock_next(32, 16, nn.GELU),
            ImprovedBlock_next(16, 8, nn.GELU),
            ImprovedBlock_next(8, 4, nn.GELU)
        ).to(device)
        self.lin_final = nn.Linear(4, 1).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        increment = self.lin_increment(rise).unsqueeze(1)
        increment = nn.LeakyReLU()(increment)

        text_hidden_states = self.pos_encoder(text_hidden_states)
        cross_text_rise = self.cross_attention(text_hidden_states, increment)
        prior_embeds = torch.nn.functional.normalize(prior_embeds, p=2.0, dim=-1)
        prior_trained = self.lin_start(prior_embeds)
        cross_text_prior = self.cross_attention(prior_trained, increment)

        concat_data = torch.concat([text_hidden_states, prior_embeds, cross_text_prior, cross_text_rise], axis=1)
        concat_data = torch.nn.functional.normalize(concat_data, p=2.0, dim=-1)

        # Apply consistent self-attention
        concat_data = self.consistent_self_attn(concat_data.permute(1, 0, 2)).permute(1, 0, 2)
        
        out = self.down_block(concat_data.permute(0, 2, 1))
        out = self.lin_final(out).permute(0, 2, 1)
        return out


class DualBranchSpliterCSA(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, max_seq_len=MAX_SEQ_LEN_K22, device='cpu'):
        super(DualBranchSpliterCSA, self).__init__()
        self.cross_attention = CrossAttentionLayer(emb_dim).to(device)
        self.pos_encoder = RotaryPositionalEmbedding(emb_dim, max_seq_len, device).to(device)
        self.consistent_self_attn = ConsistentSelfAttention(emb_dim).to(device)
        
        self.lin_increment = nn.Linear(1, emb_dim).to(device)
        self.lin_start = nn.Linear(emb_dim, emb_dim).to(device)
        self.emb_dim = emb_dim
        
        # Rise branch for handling rise-influenced data
        self.down_block_1 = nn.Sequential(
            ImprovedBlock(79, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
        ).to(device)

        self.down_block_2 = nn.Sequential(
            ImprovedBlock(154, 256, 0.3),
            ImprovedBlock(256, 128, 0.3),
            ImprovedBlock(128, 64, 0.3),
            ImprovedBlock(64, 32, 0.3),
        ).to(device)

        self.down_block_fin = nn.Sequential(
            ImprovedBlock(64, 32, 0.3),
            ImprovedBlock(32, 16, 0.3),
            ImprovedBlock(16, 1, 0.3),
        ).to(device)

    def forward(self, text_hidden_states, prior_embeds, rise):
        text_hidden_states = self.pos_encoder(text_hidden_states)

        # Base branch processing
        prior_trained = self.lin_start(prior_embeds)

        # Rise branch processing
        increment = self.lin_increment(rise).unsqueeze(1)

        concat_base = torch.concat([text_hidden_states, prior_trained, increment], axis=1)
        
        # Apply consistent self-attention
        concat_base = self.consistent_self_attn(concat_base.permute(1, 0, 2)).permute(1, 0, 2)

        cross_text_rise = self.cross_attention(text_hidden_states, increment)
        cross_text_prior = self.cross_attention(text_hidden_states, prior_trained)

        concat_cross = torch.concat([cross_text_rise, cross_text_prior], axis=1)
        
        # Apply consistent self-attention
        concat_cross = self.consistent_self_attn(concat_cross.permute(1, 0, 2)).permute(1, 0, 2)

        base_output = self.down_block_1(concat_base.permute(0, 2, 1))
        cross_output = self.down_block_2(concat_cross.permute(0, 2, 1))

        concat_out = torch.concat([base_output, cross_output], axis=-1)
        out = self.down_block_fin(concat_out).permute(0, 2, 1)
        return out


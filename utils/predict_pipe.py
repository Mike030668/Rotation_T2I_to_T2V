from step_utils.rotations import RotationVectors as RV
from utils.utills import flush_memory
import torch.nn.functional as F
import torch

class K22_T2V:
    def __init__(self, model, decoder):
        super(K22_T2V, self).__init__()
        self.model = model
        self.decoder = decoder
        self.device = decoder.device
        self.model.eval()


    def predict_step_vectors(self, start_unclip: torch.float16, init_hidden_state: torch.float16, increment: float, qty_vectors: int):
        
        last_hidden_states = torch.concat([init_hidden_state for _ in range(qty_vectors)]).to(self.device).to(torch.float32)
        base_unclip_embs = torch.concat([start_unclip.unsqueeze(dim=1) for _ in range(qty_vectors)]).to(self.device).to(torch.float32)
        deltas = torch.tensor([[increment * i] for i in range(qty_vectors)]).to(self.device).to(torch.float32)
        return self.model(last_hidden_states, base_unclip_embs, deltas)


    def predict_regress_vectors(self, start_unclip: torch.float16, hidden_state: torch.float16, text_emb: torch.float16, increment: float, qty_vectors: int, fix_text_emb=True, rote_pow=1, dim2norm=-1):
        
        deltas = torch.tensor([[increment] for _ in range(qty_vectors)]).to(self.device).to(torch.float32)
        init_hidden_state = hidden_state.to(torch.float32).to(self.device)
        init_text_emb = text_emb.to(torch.float32).to(self.device)
        init_unclip_embed = start_unclip.to(torch.float32).to(self.device)

        pred_unclip_embs = []
        with torch.no_grad():
            for i in range(len(deltas)):
                if not i:
                    pred = self.model(text_hidden_states=init_hidden_state, prior_embeds=init_unclip_embed.unsqueeze(1), rise=deltas[i].unsqueeze(1))
                    pred_unclip_embs.append(pred)
                    del pred
                    flush_memory()
                else:
                    norm_unclip_embed = F.normalize(init_unclip_embed, dim=dim2norm)
                    norm_text_emb = F.normalize(init_text_emb, dim=dim2norm)
                    cos_sim = torch.cosine_similarity(norm_unclip_embed, norm_text_emb, dim=1).to(torch.float32).to(self.device)
                    R_marixes = RV().get_rotation_matrix(init_unclip_embed, pred_unclip_embs[-1].squeeze(1))
                    roted_hidden_state = RV().cosin_rotate(init_hidden_state, cos_sim, R_marixes, power=rote_pow)

                    if not fix_text_emb:
                        init_text_emb = RV().cosin_rotate(init_text_emb, cos_sim, R_marixes, power=rote_pow)
                        init_unclip_embed = pred_unclip_embs[-1].squeeze(1)
                        init_hidden_state = roted_hidden_state

                    pred = self.model(text_hidden_states=roted_hidden_state, prior_embeds=pred_unclip_embs[-1], rise=deltas[i].unsqueeze(1))
                    pred_unclip_embs.append(pred)

                    del pred
                    flush_memory()
        return pred_unclip_embs



    def consistent_predict_step_vectors(self, start_unclip: torch.float16, init_hidden_state: torch.float16, increment: float, qty_vectors: int, batch_size: int = 5):
            
            init_hidden_state = init_hidden_state.to(torch.float32).to(self.device)
            start_unclip = start_unclip.to(torch.float32).to(self.device)
            start_delta = torch.tensor([[0]]).to(self.device).to(torch.float32)
    
            batch_hidden_states = [init_hidden_state for _ in range(batch_size)]
            batch_unclip_embeds = [start_unclip.unsqueeze(0) for _ in range(batch_size)]
            batch_deltas = [start_delta for _ in range(batch_size)]

            batch_bhs = torch.stack([hs.squeeze(0) for hs in batch_hidden_states])
            batch_bue = torch.stack([uc.squeeze(0) for uc in batch_unclip_embeds])
            batch_dlt = torch.stack([dlt.squeeze(0) for dlt in batch_deltas])

            with torch.no_grad():
              
                all_predicted_vectors = []
                # Use sliding window to predict further vectors
                for i in range(qty_vectors):
                    pred_batch = self.model(
                        text_hidden_states=batch_bhs,
                        prior_embeds=batch_bue,
                        rise=batch_dlt
                    )
                    next_vector = pred_batch[-1].unsqueeze(0)
                    all_predicted_vectors.append(next_vector)
                    next_delta = torch.tensor([[(i)*increment]]).to(self.device).to(torch.float32)

                    # Update batch: remove the first and add the last predicted vector
                    batch_deltas.pop(0)
                    batch_deltas.append(next_delta)
                    batch_dlt = torch.stack([dlt.squeeze(0) for dlt in batch_deltas])

            return all_predicted_vectors


    def consistent_predict_regress_vectors(self, start_unclip: torch.float16, hidden_state: torch.float16, text_emb: torch.float16, increment: float, qty_vectors: int, batch_size: int = 5, fix_text_emb=True, rote_pow=1, dim2norm=-1):
            
            init_hidden_state = hidden_state.to(torch.float32).to(self.device)
            start_unclip = start_unclip.to(torch.float32).to(self.device)
            start_delta = torch.tensor([[0]]).to(self.device).to(torch.float32)
            init_text_emb = text_emb.to(torch.float32).to(self.device)

            batch_hidden_states = [init_hidden_state for _ in range(batch_size)]
            batch_unclip_embeds = [start_unclip.unsqueeze(0) for _ in range(batch_size)]
            batch_deltas = [start_delta for _ in range(batch_size)]
            batch_text_embeds = [init_text_emb for _ in range(batch_size)]

            batch_bhs = torch.stack([hs.squeeze(0) for hs in batch_hidden_states])
            batch_bue = torch.stack([uc.squeeze(0) for uc in batch_unclip_embeds])
            batch_dlt = torch.stack([dlt.squeeze(0) for dlt in batch_deltas])

            init_batch_txt = torch.stack(batch_text_embeds)
            init_batch_bhs = torch.stack([hs.squeeze(0) for hs in batch_hidden_states])
            init_batch_bue = torch.stack([uc.squeeze(0) for uc in batch_unclip_embeds])

            all_predicted_vectors = []

            with torch.no_grad():

                # Use sliding window to predict further vectors
                for i in range(qty_vectors):

                    if not i:
                        pred_batch = self.model(
                            text_hidden_states=batch_bhs,
                            prior_embeds=batch_bue,
                            rise=batch_dlt
                        )

                    else:
                        norm_batch_bue = F.normalize(init_batch_bue, dim=dim2norm)
                        norm_text_emb = F.normalize(init_batch_txt, dim=dim2norm)
                        cos_sim = torch.cosine_similarity(norm_batch_bue, norm_text_emb, dim=-1).to(torch.float32).to(self.device)
 
                        R_marixes = RV().get_rotation_matrix(init_batch_bue.squeeze(1), pred_batch.squeeze(1))
                        batch_bhs = RV().cosin_rotate(init_batch_bhs, cos_sim, R_marixes, power=rote_pow)

                        if not fix_text_emb:
                            init_batch_txt = RV().cosin_rotate(init_batch_txt, cos_sim, R_marixes, power=rote_pow)
                            init_batch_bue = pred_batch
                            init_batch_bhs = batch_bhs

                        pred_batch = self.model(
                            text_hidden_states=batch_bhs,
                            prior_embeds=batch_bue,
                            rise=batch_dlt
                        )


                    next_vector = pred_batch[-1].unsqueeze(0)
                    all_predicted_vectors.append(next_vector)
                    next_delta = torch.tensor([[increment]]).to(self.device).to(torch.float32)


                    # Update batch: remove the first and add the last predicted vector
                    batch_unclip_embeds.pop(0)
                    batch_unclip_embeds.append(next_vector)
                    batch_bue = torch.stack([uc.squeeze(0) for uc in batch_unclip_embeds])

                    batch_deltas.pop(0)
                    batch_deltas.append(next_delta)
                    batch_dlt = torch.stack([dlt.squeeze(0) for dlt in batch_deltas])
 
            return all_predicted_vectors


    def generate_frames(self, set_unclips: object, negative_emb: object, height=256, width=256, num_inference_steps=50, seed=0):
        gen_frames = []
        for embeds in set_unclips:
            result = self.decoder(image_embeds=embeds.squeeze(1), negative_image_embeds=negative_emb, num_inference_steps=num_inference_steps, generator=torch.Generator().manual_seed(seed), height=height, width=width)[0][0]
            gen_frames.append(result)
            del result
            flush_memory()
        return gen_frames

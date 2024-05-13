from step_utils.rotations import RotationVectors as RV
from utils.utills import flush_memory
import torch.nn.functional as F
import torch

class K22_T2V():
    def __init__(self, model, decoder):

        super(K22_T2V, self).__init__()
        self.model = model
        self.decoder = decoder
        self.device = model.device
        self.model.model.eval()

    def predict_step_vectors(self, start_unclip: torch.float16, init_hidden_state: torch.float16, increment:float, qty_vectors:int):

        last_hidden_states = torch.concat([init_hidden_state for _ in range(qty_vectors)]).to(self.device).to(torch.float32)
        base_unclip_embs = torch.concat([start_unclip.unsqueeze(dim=1) for _ in range(qty_vectors)]).to(self.device).to(torch.float32)
        deltas = torch.tensor([[increment*i] for i in range(qty_vectors)]).to(self.device).to(torch.float32)
        return self.model.model(last_hidden_states, base_unclip_embs, deltas)



    def predict_regress_vectors(self, start_unclip: torch.float16, hidden_state: torch.float16,
                                text_emb: torch.float16, increment:float, qty_vectors:int,
                                rote_pow = 1, fix_text_emb = True, dim2norm = -1
                                ):


        deltas = torch.tensor([[increment] for _ in range(qty_vectors)]).to(self.device).to(torch.float32)
        init_hidden_state = hidden_state.to(torch.float32).to(self.device)
        init_text_emb = text_emb.to(torch.float32).to(self.device)
        init_unclip_embed = start_unclip.to(torch.float32).to(self.device)

        pred_unclip_embs =[]
        with torch.no_grad(): # prepare data for model and losses with no_grad

            for i in range(len(deltas)):
              if not i:

                  pred = self.model.model(
                                text_hidden_states = init_hidden_state,
                                prior_embeds = init_unclip_embed.unsqueeze(1),
                                rise = deltas[i].unsqueeze(1),
                                )

                  pred_unclip_embs.append(pred)
                  del(pred)
                  flush_memory()


              else:
                  norm_unclip_embed =  F.normalize(init_unclip_embed, dim = dim2norm )
                  norm_text_emb =  F.normalize(init_text_emb, dim = dim2norm )
                  cos_sim = torch.cosine_similarity(norm_unclip_embed, norm_text_emb, dim = 1).to(torch.float32).to(self.device)

                  R_marixes = RV().get_rotation_matrix(init_unclip_embed, pred_unclip_embs[-1].squeeze(1))

                  roted_hidden_state = RV().cosin_rotate(init_hidden_state,
                                                        cos_sim,
                                                        R_marixes,
                                                        power = rote_pow
                                                        )


                  if not fix_text_emb:
                      init_text_emb = RV().cosin_rotate(init_text_emb,
                                                        cos_sim,
                                                        R_marixes,
                                                        power = rote_pow
                                              )
                      init_unclip_embed  =  pred_unclip_embs[-1].squeeze(1)

                      init_hidden_state = roted_hidden_state


                  pred = self.model.model(
                                      text_hidden_states = roted_hidden_state,
                                      prior_embeds = pred_unclip_embs[-1],
                                      rise = deltas[i].unsqueeze(1),
                                      )

                  pred_unclip_embs.append(pred)
                  del(pred)
                  flush_memory()

        return pred_unclip_embs

    def generate_frames(self, set_unclips: object,  negative_emb: object, height = 256, width = 256, num_inference_steps = 50, seed = 0):

        generator = torch.Generator().manual_seed(seed)
        gen_frames = []
        # generate from latent #####################################################
        for embeds in set_unclips:
            result = self.decoder(
                                image_embeds = embeds.squeeze(1),
                                negative_image_embeds = negative_emb,
                                num_inference_steps=num_inference_steps,
                                generator = generator,
                                height=height,
                                width=width)[0][0]


            gen_frames.append(result)
            del(result)
            flush_memory()

        return gen_frames
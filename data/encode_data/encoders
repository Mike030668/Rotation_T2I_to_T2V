from diffusers import KandinskyV22PriorPipeline
import PIL
from transformers import CLIPVisionModelWithProjection
import torch
import pandas as pd
from tqdm.notebook import tqdm


class Encode_Kand22():
      def __init__(self,
                 device: str,
                 model_name = 'kandinsky-community/kandinsky-2-2-prior',
                 ):
        
            super(Encode_Kand22, self).__init__()

            self.TYPE_TENSORS = torch.float16
            self.DEVICE = device

            image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name,
                                                                  subfolder='image_encoder'
            ).to(self.TYPE_TENSORS).to(self.DEVICE)

            self.prior_T2I = KandinskyV22PriorPipeline.from_pretrained(model_name,
                                                                  image_encoder=image_encoder,
                                                                  torch_dtype=self.TYPE_TENSORS
            )

            self.prior_tokenizer = self.prior_T2I.tokenizer

            self.prior_T2I.to(self.DEVICE)

            del image_encoder
            self.flush_memory()


      def get_image_embeds(self, pil_img, to_square = None, recize = []):
    
            if to_square:
                width, height = pil_img.size   # Get dimensions
                square_size = min(width, height)

                left = (width - square_size)/2
                top = (height - square_size)/2
                right = (width + square_size)/2
                bottom = (height + square_size)/2

                # Crop the center of the image
                pil_img.crop((left, top, right, bottom))

            if len(recize):
                pil_img = pil_img.resize(recize, PIL.Image.BICUBIC)

            with torch.no_grad():
                inputs = self.prior_T2I.image_processor(pil_img,
                                                return_tensors="pt")
                outputs = self.prior_T2I.image_encoder(**inputs.to(self.DEVICE)).image_embeds
                del inputs
                self.flush_memory()
            return outputs


      def get_unclip_embeds(self, caption, num_steps_inf = 30, seed = 0):
            return self.prior_T2I(prompt=caption,
                                num_inference_steps=num_steps_inf,
                                num_images_per_prompt=1,
                                generator =  torch.manual_seed(seed)
                                ).image_embeds


      def get_text_embeds(self, caption):
          
            inputs = self.prior_tokenizer([caption],
                                        padding="max_length",
                                        max_length=self.prior_tokenizer.model_max_length,
                                        truncation=True,
                                        return_tensors="pt"
                                        )
            
            text_embeds = self.prior_T2I.text_encoder(**inputs.to(self.DEVICE)).text_embeds

            
            last_hidden_states = self.prior_T2I.text_encoder(**inputs).last_hidden_state

            return text_embeds, last_hidden_states


      def flush_memory(self):
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            with torch.no_grad():
                for _ in range(3):
                  torch.cuda.empty_cache()
                  torch.cuda.ipc_collect()

      def encodedata(self,
                     height: int,
                     width: int,
                     pil_df : pd.DataFrame,
                     max_farme: int,
                     path_to : str,
                     save_each = 25
                     ):
          
          pil_ids_movi_unique = pil_df.id_movi.unique()
          all_data = []

          with torch.no_grad(): # prepare data for model and losses with no_grad
              for id_movi in tqdm(pil_ids_movi_unique, unit = " movi "):
                  
                  frames_pathes = pil_df[pil_df.id_movi == id_movi].paths
                  frames_ids = pil_df[pil_df.id_movi == id_movi].id_frame.values

                  qty_frames = len(frames_ids)

                  if qty_frames > max_farme:
                      # for normal way take before last
                      frames_pathes = frames_pathes[:max_farme]

                  image_embeds = []
                  ids_frame = []

                  for i, path in enumerate(frames_pathes):
                      
                      pil_image = PIL.Image.open(path)
                      image_embeds.append(self.get_image_embeds(pil_image,
                                                  to_square=True,
                                                  recize=(height, width)).cpu())
                      # take label frames
                      ids_frame.append(int(frames_ids[i]))


                  if len(ids_frame):
                      caption = pil_df[pil_df.id_movi == id_movi].caption_movi.values[0]

                      text_embeds, last_hidden_states = self.get_text_embeds(caption)

                      dict_movi = dict()
                      dict_movi['id_movi'] = id_movi
                      dict_movi['img_embeds'] = image_embeds
                      dict_movi['text_embed'] = text_embeds.cpu()
                      dict_movi['last_hidden_state'] = last_hidden_states.cpu()
                      dict_movi['unclip_embed'] = self.get_unclip_embeds.cpu()
                      dict_movi['ids_frames'] = ids_frame
                      del(image_embeds, inputs)
                      self.flush_memory()
                      all_data.append(dict_movi)

                  if len(all_data) and not len(all_data) % save_each:
                     torch.save(all_data, path_to)

              torch.save(all_data, path_to)
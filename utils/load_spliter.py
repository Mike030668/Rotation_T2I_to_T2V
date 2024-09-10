import gdown
import torch

class Splitter_K22():
  def __init__(self, name_script, name_model, device='cuda'):
    super(Splitter_K22, self).__init__()
    self.device = device
    self.name_script = name_script
    self.name_model = name_model

  
  def init_spliter(self):
      import_str = f"from build_models.{self.name_script} import {self.name_model}"
      script_spliter = dict()
      exec(import_str,  script_spliter)
      # init_spliter
      self.model = script_spliter[self.name_model](device=self.device).to(self.device)


  def load_weights(self, name_weights, id_weights, dir = "/content"):
      file_name = f"{name_weights}.cpt"
      gdown.download('https://drive.google.com/uc?id=' + id_weights, file_name, quiet=False)
      try:
        self.model.load_state_dict(torch.load(f"{dir}/{file_name}"))
      except:
        checkpoint = torch.load(f"{dir}/{file_name}")
        self.model.load_state_dict(checkpoint['model_state_dict'])

import os
def compare_checkpoints(list_model_1, list_model_2, dir = "/content"):
    # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    # take weights from both checkpoints
    """
    :param list_model_1 
    _________
    dict_ckpts =   { "Spliter_next_base_loss_pw_0_tgif300_1":   {"loss_62": "1jkaET5U-ziAVYRZZX5B1v7tlGBwFbUgL"
                    },
                   "Increament_next_Mixloss_ab_1_pw_1_acc_norm_minus_1_300_500pxls_new_all_ways_loss":   {"mse_15": "1j6Kw-uT2-w1s-ZFDe18JCLww1zg1g2-W"
                    }
                    )
    type_train = "Spliter_next_base_loss_pw_0_tgif300_1"
    type_cpt = "loss_62"
    file_id = dict_ckpts[type_train][type_cpt]
    list_model_1 = [type_cpt, file_id]
    ....
    """
    name_1, id_weights_1 = list_model_1[0], list_model_1[1]
    file_name_1 = f"{name_1}.cpt"
    gdown.download('https://drive.google.com/uc?id=' + id_weights_1, file_name_1, quiet=False)
    weights_1 = torch.load(file_name_1, weights_only=False)["model_state_dict"]



    name_2, id_weights_2 = list_model_2[0], list_model_2[1]
    file_name_2 = f"{name_2}.cpt"
    gdown.download('https://drive.google.com/uc?id=' + id_weights_2, file_name_2, quiet=False)
    weights_2 = torch.load(file_name_2, weights_only=False)["model_state_dict"]

    models_differ = 0
    all_diff = 0
    print()
    for key_item_1, key_item_2 in zip(weights_1.items(), weights_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):

                diff = (weights_1[key_item_1[0]] - weights_2[key_item_2[0]]).cpu().abs().numpy().sum()
                print(f'Found difference {diff:.5e} at layer  {key_item_1[0]}')
                all_diff+=diff
            else:
                raise Exception
    print()            
    print(f'All difference in moddel {all_diff:.5e}')
    if models_differ == 0:
        print('Models are same! :)')

    del(weights_1, weights_2)
    os.remove(f"{dir}/{file_name_1}")
    os.remove(f"{dir}/{file_name_2}")
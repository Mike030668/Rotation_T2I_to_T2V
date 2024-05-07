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
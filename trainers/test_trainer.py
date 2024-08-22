import os
import random
import numpy as np
import torch
from tqdm.notebook import tqdm


from step_utils.step_points import Shuff_Reshuff


class ExpandTrainer():
        """
        Expand trainer class with possibility to make
        several autoregress steps including consistent splitter models
        """
        def __init__(self,
                      model,
                      device,
                      path_save: str,
                      next_train = False,
                      path_cpt = None,

                      ):
              super(ExpandTrainer, self).__init__()
              self.path_save = path_save
              self.next_train = next_train
              self.DEVICE = device
              self.model = model
              self.emb = model.emb_dim
              self.model_class = self.model.__class__.__name__

              if self.next_train:
                  self.checkpoint = torch.load(path_cpt)
                  self.model.load_state_dict(self.checkpoint['model_state_dict'])
                  self.last_save = self.checkpoint['saved_model']
                  self.last_epoch = self.checkpoint['epoch']
                  self.last_lr = self.checkpoint["last_history"]["lr"][-1] #
                  self.best_loss = self.checkpoint['loss']
                  self.best_loss_diff = self.checkpoint['diff_loss']
                  self.best_loss_rote = self.checkpoint['rote_loss']
                  self.best_acc = self.checkpoint['acc']
                  self.history_train = self.checkpoint['all_history']
                  self.best_eph = {"loss" : self.last_epoch,
                                  "rote" : self.last_epoch,
                                  "diff" : self.last_epoch,
                                  "acc" : self.last_epoch,
                                  }
                  self.last_checkpoint =  str(path_cpt.split("/")[-1])

              # state for start
              else:
                  self.last_checkpoint = "New"
                  self.last_save = ''
                  self.last_epoch = 0
                  self.best_loss = 1000000
                  self.best_loss_diff = 1000000
                  self.best_loss_rote = 1000000
                  self.best_acc = 0
                  self.best_eph = {"loss" : 0,
                                  "rote" : 0,
                                  "diff" : 0,
                                  "acc" : 0,
                                  }

                  self.history_train = {"loss" : [],
                                        "loss_rote" : [],
                                        "loss_diff" : [],
                                        "acc" : [],
                                        "lr" : [],
                                        "base_loss" : []
                                        }

                  self.best_eph = {"loss" : 0,
                                    "rote" : 0,
                                    "diff" : 0,
                                    "acc" : 0
                                    }

              self.hist = {"loss" : [],
                          "loss_rote" : [],
                          "loss_diff" : [],
                          "acc" : [],
                          "base_loss" : [],
                          "lr" : []
                  }

              self.logs = {"bad_movi" : [],
                          "empty" : [],
                          "naninf_loss" : [],
                          "naninf_diff" : [],
                          "naninf_acc" : [],
                          "naninf_rote" : []
                  }

              # paths to save temory weights
              PATH_model_loss = self.path_save +'/last_best_loss.pt'
              PATH_model_rote = self.path_save +'/last_best_rote.pt'
              PATH_model_diff = self.path_save +'/last_best_diff.pt'
              PATH_model_acc = self.path_save +'/last_best_acc.pt'

              foder_temp_checkpoints = self.path_save + "/temp_checkpoints"
              if not os.path.exists(foder_temp_checkpoints):
                   os.makedirs(foder_temp_checkpoints)

              self.dict_model_paths = {"loss" : PATH_model_loss,
                                        "rote" : PATH_model_rote,
                                        "diff" : PATH_model_diff,
                                        "acc" : PATH_model_acc,
                                        "temp": foder_temp_checkpoints
                                        }

              # init waitings jdun
              self.wait_train = 0
              self.wait2end = 0

        def trailoop(self,
                     train_data,
                     max_batch : int,
                     optimizer: object,
                     scheduler: object,
                     maker_points: object,
                     rotator: object,
                     combine_loss:object,
                     cust_accuracy: object,
                     epochs: int,
                     friq_save_checkpoint = 15,
                     jdun_train = 3,
                     jdun_end = 3,
                     use_on_epoch = .9,
                     exp_rate = 0.99,
                     window = 3,
                     update_best = 0,
                     learning_rate = 1e-03,
                     update_lr = False,
                     reset_losses_acc = False,
                     suff_direct =  False,
                     add_back_train = False,
                     add_rote_train = False,
                     add_diff_train = False,
                     consistent =  False,
                     pow_rote = 1,
                     clip_grad_norm = False,
                     change_name_to_save = None,
                     ):

            # correct save, train and lerning rate
            self.suff_direct =  suff_direct
            self.add_back_train = add_back_train
            self.add_rote_train = add_rote_train

            if add_diff_train == True: self.add_diff_train = 1
            else: self.add_diff_train = add_diff_train

            self.consistent =  consistent
            if consistent or self.add_diff_train > 1:
              print("consistent == True or add_diff_train, in result suff_direct will False")
              self.suff_direct = False

            self.pow_rote = pow_rote
            self.fr_sv_cpt = friq_save_checkpoint
            self.window = window
            self.JDUN_END = jdun_end
            self.JDUN_TRAIN = jdun_train
            self.max_batch = max_batch
            self.use_on_epoch = use_on_epoch
            self.train_data = train_data
            if epochs < max(epochs, window+2):
                print(f"EPOCHS have to >= window+2, chahged onto {window+2}")
            self.EPOCHS = max(epochs, window+2)
            self.update_lr = update_lr
            self.LR_RATE = learning_rate
            self.EXP_RATE = exp_rate
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.GAMMA = self.scheduler.gamma
            self.reset_losses_acc = reset_losses_acc
            self.change_name_to_save = change_name_to_save

            # Custom loss and acc
            self.cos_acc_rote = cust_accuracy
            self.combi_loss = combine_loss

            # ComputeDiffPoints class
            self.maker_points = maker_points

            # inite rotation class
            self.RV = rotator()

            # wait to update_best
            self.update_best = update_best + self.window

            # ways in train
            ways = 2 if self.add_back_train else 1
            type_ways = ["normal", "back"]

            # correction for next_train
            if self.next_train:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.LR_RATE  if self.update_lr else self.last_lr
                self.scheduler.last_epoch = self.last_epoch

                if self.reset_losses_acc:
                      self.last_save = ''
                      self.last_epoch = 0
                      self.best_loss = 1000000
                      self.best_loss_diff = 1000000
                      self.best_loss_rote = 1000000
                      self.best_acc = 0
                      self.best_eph = {"loss" : 0,
                                      "rote" : 0,
                                      "diff" : 0,
                                      "acc" : 0,
                                      }



            for epoch  in tqdm(range(self.last_epoch, self.last_epoch + self.EPOCHS),
                               unit ="EPOHS", desc ="Пробегаемся по всем эпохам"):
                # inite
                self.flush_memory()
                text = ''
                save_model = 0
                eph_loss = 0
                eph_loss_rote = 0
                eph_loss_diff = 0
                eph_cos_acc = 0
                eph_base_loss = 0
                cos_acc = 0
                cur_lr = self.optimizer.param_groups[0]['lr']

                random.shuffle(self.train_data)  # shuffle  data each epoch
                take = int(len(self.train_data)*self.use_on_epoch)
                take_data = self.train_data[:take]  # take shuffle part data each epoch


                # go by selected data
                for  id_m, data  in tqdm(enumerate(take_data), unit = f" movi ",
                                                    desc ="Пробегаемся по всем фильмам"):
                    self.flush_memory()
                    # get next frame embbedings
                    id_movi, ids_frame = data['id_movi'], data['ids_frames']
                    text_hid_state = data['last_hidden_state']
                    text_embed = data['text_embed']
                    unclip_embed = data['unclip_embed']
                    image_embeds = data['img_embeds']

                    # qty_frames used for normalization steps and control long movi too
                    qty_frames = len(image_embeds)

                    # control long movi
                    d_batch = self.max_batch-1
                    if qty_frames and id_movi:
                        if qty_frames <= self.max_batch:
                            d_batch = qty_frames - 1

                        # if used model with consistent_attention
                        if self.consistent or self.add_diff_train > 1:
                          s_id = np.random.randint(1, qty_frames - d_batch + 1)
                          # make set time_labels
                          rand_ids = [0] + list(np.arange(s_id, s_id + d_batch))

                        else:
                          # take random ids in ids_frame[1:] because ids_frame[0] will first
                          rand_ids = [0] + list(np.random.choice(np.arange(1, qty_frames-1),
                                                                  d_batch-1, replace=False))
                          rand_ids = sorted(rand_ids + [qty_frames - 1])  # Sort the selected indices

                        # select rand_ids labels and image_embeds
                        image_embeds = torch.concat([image_embeds[i] for i in rand_ids]).unsqueeze(dim=1)

                        # collect unclip tensors
                        base_unclip_embs = [unclip_embed for _ in range(d_batch)]
                        base_unclip_embs = torch.concat(base_unclip_embs).unsqueeze(dim=1)#

                        # collect text_hid_states tensors
                        text_hid_states = [text_hid_state for _ in range(d_batch)]
                        text_hid_states =  torch.concat(text_hid_states)

                        # collect text tensors
                        text_embs = [text_embed for _ in range(d_batch)]
                        text_embs =  torch.concat(text_embs).unsqueeze(dim=1)

                        labels = np.array([ids_frame[i] for i in rand_ids])

                        # place labels to class points
                        self.maker_points.time_labels = labels
                        config_norm, config_back = self.maker_points.getpoints_rotetrain()


                        for way in range(ways):

                            way_loss = 0
                            way_rote_loss = 0
                            way_diff_loss = 0
                            way_loss_base = 0

                            if not way: # normal way
                                all_points = self.maker_points.points

                            if way and self.add_back_train: # back way
                                all_points = (-1) * self.maker_points.back_points


                            # intit class for shufflee
                            b_srs = Shuff_Reshuff(len(all_points))
                            if self.suff_direct:
                                bs_idx = b_srs.shuffle()   # shuffleed indexees
                                bu_idx = b_srs.unshuffle() # indexees for re_shuffle
                            else:
                                bs_idx = b_srs.idx_base # shuffleed indexees
                                bu_idx = b_srs.idx_base # indexees for re_shuffle

                            ######### compute enters and states for model and losses for base ways

                            # collect time directions tensors and normalize
                            increments_base = torch.tensor(all_points/qty_frames).unsqueeze(1)

                            if not way: # normal way
                                # collect base_img_embs tensors
                                base_img_embs = [image_embeds[0] for _ in range(d_batch)]

                            if way and self.add_back_train: # back way
                                base_img_embs = [image_embeds[-1] for _ in range(d_batch)]

                            base_img_embs = torch.concat(base_img_embs).unsqueeze(dim=1)


                            if not way: # normal way
                                # collect img_embs tensors
                                img_embs = image_embeds[1:]

                            if way and self.add_back_train: # back way
                                # collect back_img_embs tensors
                                img_embs = torch.flip(image_embeds, [0,])[1:]

                            ####################################################################

                            # img difference
                            diff_img_embs =  (base_img_embs[bs_idx].squeeze(dim=1) - img_embs[bs_idx].squeeze(dim=1)).to(torch.float32) #

                            # check on static movi
                            bad_movi = 0
                            for k in range(diff_img_embs.shape[0]):
                              if abs(diff_img_embs[k].sum()) == 0:
                                bad_movi+=1

                            if not bad_movi:
                                # zero_grad optimizer
                                self.optimizer.zero_grad()

                                # base predict which can used in next step
                                pred_unclip_embs = self.model(
                                  text_hidden_states = text_hid_states[bs_idx].to(torch.float32).to(self.DEVICE), # shuffleed
                                  prior_embeds = base_unclip_embs[bs_idx].to(torch.float32).to(self.DEVICE),
                                  rise = increments_base[bs_idx].to(torch.float32).to(self.DEVICE)
                                                    )

                                # difference
                                rote_loss, diff_loss = self.combi_loss(base_img_embs[bs_idx], img_embs[bs_idx],
                                                                      base_unclip_embs[bs_idx], pred_unclip_embs)
                                batch_size = len(bs_idx)
                                way_rote_loss += rote_loss/batch_size
                                way_diff_loss += diff_loss/batch_size

                                # control NAN INF in loss
                                if torch.isnan(way_rote_loss).sum() or torch.isinf( way_rote_loss).sum():
                                  print(f"\rMovi {id_movi} rote_loss base_step isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_rote"].append(id_movi)
                                  break


                                # control NAN INF in loss
                                if torch.isnan(way_diff_loss).sum() or torch.isinf(way_diff_loss).sum():
                                  print(f"\rMovi {id_movi} diff_loss base_step isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_diff"].append(id_movi)
                                  break

                                # collect base_loss
                                way_loss_base += (rote_loss + diff_loss)/batch_size # (0) for 1280
                                eph_base_loss += way_loss_base.mean().item()


                                cos_acc =   self.cos_acc_rote(init_img_vec = base_img_embs[bs_idx],
                                                              next_img_vec = img_embs[bs_idx],
                                                              init_unclip = base_unclip_embs[bs_idx],
                                                              pred_unclip = pred_unclip_embs,
                                                              )
                                 # collect cos_acc
                                eph_cos_acc+= cos_acc.mean().item()

                                # control NAN INF in acc
                                if torch.isnan(cos_acc).sum() or torch.isinf(cos_acc).sum():
                                  print(f"\rMovi {id_movi} cos_acc_base isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_acc"].append(id_movi)
                                  break

                                # temp show losses for batch
                                if id_m:
                                  rate = id_m*ways
                                  print(f"\rMovi {id_movi} step {id_m} base_{type_ways[way]}_way diff {eph_loss_diff/rate:.5f} | rote {eph_loss_rote/rate:.5f} | lr {cur_lr:.5e}", end="")

                                #########  Rotation train steps ########################################
                                to_rote, rote_norm, rote_back = 0, 0, 0

                                if self.add_rote_train:#
                                    # Rotation train steps
                                    rote_norm = len(config_norm['id_uclip_emb'])

                                    if self.add_back_train:
                                        rote_back = len(config_back['id_uclip_emb'])

                                    # control size batch
                                    if not way:
                                      id_img_emb_s = np.array(config_norm['id_img_emb_s'])
                                      id_uclip_emb = np.array(config_norm['id_uclip_emb'])
                                      id_img_delta = np.array(config_norm['id_img_delta'])
                                      norm_delta = np.array(config_norm['norm_delta'])


                                    if way and rote_back:
                                      id_img_emb_s = np.array(config_back['id_img_emb_s'])
                                      id_uclip_emb = np.array(config_back['id_uclip_emb'])
                                      id_img_delta = np.array(config_back['id_img_delta'])
                                      norm_delta = (-1)*np.array(config_back['norm_delta'])

                                    to_rote =  rote_norm + rote_back

                                if self.add_rote_train and to_rote: #
                                    unique_id_img_emb = np.unique(id_img_emb_s)
                                    steck_idx = []
                                    for unique_id in unique_id_img_emb:
                                            # take indes for eacc rote step
                                            set_dxs = np.where(id_img_emb_s == unique_id)[0]
                                            # collect steck
                                            steck_idx.append(set_dxs)

                                    if not self.consistent:
                                        rote_steps = len(unique_id_img_emb)

                                        for set_dxs in steck_idx:
                                            # intit class for shufflee again
                                            srs = Shuff_Reshuff(len(set_dxs))
                                            if self.suff_direct:
                                                s_idx = srs.shuffle() # shuffleed indexees
                                            else:
                                                s_idx = srs.idx_base  # shuffleed indexees

                                            # collect norm steps
                                            take_text_hid_states = torch.clone(text_hid_states)[id_uclip_emb[set_dxs],:,:]
                                            take_base_unclip_embs = torch.clone(base_unclip_embs)[id_uclip_emb[set_dxs],:,:]
                                            take_base_img_embs = torch.clone(base_img_embs)[id_uclip_emb[set_dxs],:,:]
                                            take_text_embs = torch.clone(text_embs)[id_uclip_emb[set_dxs],:,:]
                                            base_img_embs_2rt = torch.clone(img_embs)[id_img_emb_s[set_dxs],:,:]
                                            image_embs_2rt = torch.clone(img_embs)[id_img_delta[set_dxs],:,:]
                                            increments_2rt = torch.tensor(norm_delta[set_dxs]).unsqueeze(1)

                                            # shufle
                                            take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                            take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                            take_base_img_embs = take_base_img_embs[s_idx].to(self.DEVICE)
                                            take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                            base_img_embs_2rt = base_img_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                            image_embs_2rt = image_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                            increments_2rt = increments_2rt[s_idx].to(self.DEVICE)


                                            # get rotation marixes_i2i
                                            R_marixes_i2i = self.RV.get_rotation_matrix(take_base_img_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                        base_img_embs_2rt.squeeze(1).to(torch.float32)).to(self.DEVICE)

                                            # get cos_sim  base_img_embs vectors and base_unclip_embs vectors
                                            cos_sim_1 = torch.cosine_similarity(take_base_img_embs.to(self.DEVICE), take_base_unclip_embs, dim = -1)

                                            # compute roted unclip_embs with R_marixes_i2i and cos_sim base_img_embs and base_unclip_embs
                                            unclip_embs_2rt = self.RV.cosin_rotate(take_base_unclip_embs, cos_sim_1.to(self.DEVICE), R_marixes_i2i,  power = self.pow_rote)

                                            # get rotation marixes_u2u
                                            R_marixes_u2u = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                        unclip_embs_2rt.squeeze(1).to(torch.float32))

                                            cos_sim_2 = torch.cosine_similarity(take_base_unclip_embs.to(self.DEVICE), take_text_embs, dim = -1)
                                            text_hid_states_2rt = self.RV.cosin_rotate(take_text_hid_states, cos_sim_2.to(self.DEVICE), R_marixes_u2u, power = self.pow_rote)


                                            # rotation predict
                                            pred_rote_embs = self.model(
                                                          text_hidden_states = text_hid_states_2rt,
                                                          prior_embeds = unclip_embs_2rt,
                                                          rise = increments_2rt.to(torch.float32)
                                                                            )

                                            # combi_loss
                                            rote_loss, diff_loss = self.combi_loss(base_img_embs_2rt, image_embs_2rt,
                                                                                  unclip_embs_2rt, pred_rote_embs)
                                            batch_size = len(s_idx)
                                            way_rote_loss += rote_loss/(batch_size*rote_steps)
                                            way_diff_loss += diff_loss/(batch_size*rote_steps)

                                            print(f"\rway_rote_loss {way_rote_loss.mean().item()}, way_diff_loss {way_diff_loss.mean().item()}", end="")

                                            # control NAN INF in loss
                                            if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                              print(f"\rMovi {id_movi} rote_loss rote_step isnan or isinf, {type_ways[way]}_way")
                                              self.logs["naninf_rote"].append(id_movi)
                                              break

                                            # control NAN INF in loss
                                            if torch.isnan(way_diff_loss).sum() or torch.isinf(way_diff_loss).sum():
                                              print(f"\rMovi {id_movi} diff_loss rote_step isnan or isinf, {type_ways[way]}_way")
                                              self.logs["naninf_diff"].append(id_movi)
                                              break

                                            del(increments_2rt, take_text_embs, take_base_unclip_embs, take_base_img_embs, take_text_hid_states)
                                            del(pred_rote_embs, text_hid_states_2rt, R_marixes_i2i, R_marixes_u2u)
                                            del(unclip_embs_2rt, image_embs_2rt)
                                            self.flush_memory()

                                    elif self.consistent:
                                        # make new steck id in vertical formate for new rote predictions
                                        new_steck_idx = [[] for _ in range(len(unique_id_img_emb))]
                                        for i in range(len(steck_idx)):
                                          for idxs in steck_idx:
                                            if i in range(len(idxs)):
                                              new_steck_idx[i].append(idxs[i])

                                        rote_steps = len(new_steck_idx)

                                        for set_dxs in new_steck_idx:

                                            # intit class for shufflee again
                                            srs = Shuff_Reshuff(len(set_dxs))
                                            if self.suff_direct:
                                                s_idx = srs.shuffle() # shuffleed indexees
                                            else:
                                                s_idx = srs.idx_base  # shuffleed indexees

                                            # collect norm steps
                                            take_text_hid_states = torch.clone(text_hid_states)[id_uclip_emb[set_dxs],:,:]
                                            take_base_unclip_embs = torch.clone(base_unclip_embs)[id_uclip_emb[set_dxs],:,:]
                                            take_base_img_embs = torch.clone(base_img_embs)[id_uclip_emb[set_dxs],:,:]
                                            take_text_embs = torch.clone(text_embs)[id_uclip_emb[set_dxs],:,:]
                                            base_img_embs_2rt = torch.clone(img_embs)[id_img_emb_s[set_dxs],:,:]
                                            image_embs_2rt = torch.clone(img_embs)[id_img_delta[set_dxs],:,:]
                                            increments_2rt = torch.tensor(norm_delta[set_dxs]).unsqueeze(1)

                                            # shufle
                                            take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                            take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                            take_base_img_embs = take_base_img_embs[s_idx].to(self.DEVICE)
                                            take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                            base_img_embs_2rt = base_img_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                            image_embs_2rt = image_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                            increments_2rt = increments_2rt[s_idx].to(self.DEVICE)


                                            # get rotation marixes_i2i
                                            R_marixes_i2i = self.RV.get_rotation_matrix(take_base_img_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                        base_img_embs_2rt.squeeze(1).to(torch.float32)).to(self.DEVICE)

                                            # get cos_sim  base_img_embs vectors and base_unclip_embs vectors
                                            cos_sim_1 = torch.cosine_similarity(take_base_img_embs.to(self.DEVICE), take_base_unclip_embs, dim = -1)

                                            # compute roted unclip_embs with R_marixes_i2i and cos_sim base_img_embs and base_unclip_embs
                                            unclip_embs_2rt = self.RV.cosin_rotate(take_base_unclip_embs, cos_sim_1.to(self.DEVICE), R_marixes_i2i,  power = self.pow_rote)

                                            # get rotation marixes_u2u
                                            R_marixes_u2u = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                        unclip_embs_2rt.squeeze(1).to(torch.float32))

                                            cos_sim_2 = torch.cosine_similarity(take_base_unclip_embs.to(self.DEVICE), take_text_embs, dim = -1)
                                            text_hid_states_2rt = self.RV.cosin_rotate(take_text_hid_states, cos_sim_2.to(self.DEVICE), R_marixes_u2u, power = self.pow_rote)


                                            # rotation predict
                                            pred_rote_embs = self.model(
                                                          text_hidden_states = text_hid_states_2rt,
                                                          prior_embeds = unclip_embs_2rt,
                                                          rise = increments_2rt.to(torch.float32)
                                                                            )

                                            # combi_loss
                                            rote_loss, diff_loss = self.combi_loss(base_img_embs_2rt, image_embs_2rt,
                                                                                  unclip_embs_2rt, pred_rote_embs)
                                            batch_size = len(s_idx)
                                            way_rote_loss += rote_loss/(batch_size*rote_steps)
                                            way_diff_loss += diff_loss/(batch_size*rote_steps)
                                            print(f"\rvertical_way_rote_loss {way_rote_loss.mean().item()}, way_diff_loss {way_diff_loss.mean().item()}", end="")

                                            # control NAN INF in loss
                                            if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                              print(f"\rMovi {id_movi} rote_loss_rote isnan or isinf, {type_ways[way]}_way")
                                              self.logs["naninf_rote"].append(id_movi)
                                              break

                                            # control NAN INF in loss
                                            if torch.isnan(way_diff_loss).sum() or torch.isinf(way_diff_loss).sum():
                                              print(f"\rMovi {id_movi} diff_loss_rote isnan or isinf, {type_ways[way]}_way")
                                              self.logs["naninf_diff"].append(id_movi)
                                              break

                                            del(increments_2rt, take_text_embs, take_base_unclip_embs, take_base_img_embs, take_text_hid_states)
                                            del(pred_rote_embs, text_hid_states_2rt, R_marixes_i2i, R_marixes_u2u)
                                            del(unclip_embs_2rt, image_embs_2rt)
                                            self.flush_memory()

                                #########  Diff train steps ########################################
                                # Diff train steps
                                if self.add_diff_train:

                                    for diff_step in range(1, self.add_diff_train+1):
                                        # method for several autoregress steps
                                        config_diff_norm, config_diff_back = self.maker_points.getpoints_Ndiftrain(diff_step)
                                        #config_diff_norm, config_diff_back = self.maker_points.getpoints_diftrain()
                                        diff_norm, diff_back = 0, 0

                                        if not way:
                                          diff_norm = len(config_diff_norm['id_uclip_emb'])
                                          id_img_emb_s = np.array(config_diff_norm['id_img_emb_s'])
                                          id_uclip_emb = np.array(config_diff_norm['id_uclip_emb'])
                                          id_img_delta = np.array(config_diff_norm['id_img_delta'])
                                          norm_delta = np.array(config_diff_norm['norm_delta'])

                                        if way:
                                          diff_back = len(config_diff_back['id_uclip_emb'])
                                          id_img_emb_s = np.array(config_diff_back['id_img_emb_s'])
                                          id_uclip_emb = np.array(config_diff_back['id_uclip_emb'])
                                          id_img_delta = np.array(config_diff_back['id_img_delta'])
                                          norm_delta = (-1)*np.array(config_diff_back['norm_delta'])


                                        if (not way and diff_norm) or (way and diff_back):
                                           # print(f"way_{way}")

                                            if diff_step == 1:
                                                # un_shuffleed
                                                take_pred_unclip_embs = torch.clone(pred_unclip_embs.detach().cpu())[bu_idx]

                                            if diff_step > 1:
                                                # un_shuffleed
                                                take_pred_unclip_embs = torch.clone(next_pred_unclip_embs.detach().cpu())[su_idx]

                                            # intit class for shufflee again
                                            srs = Shuff_Reshuff([diff_norm, diff_back][way])
                                            if self.suff_direct:
                                                s_idx = srs.shuffle() # shuffleed indexees
                                                # update bu_idx for new diff_diff
                                                su_idx = srs.unshuffle() # indexees for re_shuffle
                                            else:
                                                s_idx = srs.idx_base # shuffleed indexees
                                                # update bu_idx for new diff_diff
                                                su_idx = srs.idx_base # indexees for re_shuffle


                                            # collect diff norm steps
                                            take_base_unclip_embs = torch.clone(base_unclip_embs)[id_uclip_emb,:,:]
                                            take_text_hid_states = torch.clone(text_hid_states)[id_uclip_emb,:,:]
                                            take_text_embs = torch.clone(text_embs)[id_uclip_emb,:,:]
                                            next_unclip_embs = torch.clone(take_pred_unclip_embs)[id_uclip_emb,:,:]


                                            next_base_img_embs = torch.clone(img_embs)[id_img_emb_s,:,:]
                                            next_image_embs = torch.clone(img_embs)[id_img_delta,:,:]
                                            next_increments = torch.tensor(norm_delta).unsqueeze(1)


                                            # collect torch.concat
                                            take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                            take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                            take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                            next_unclip_embs = next_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                            next_base_img_embs = next_base_img_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                            next_image_embs = next_image_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                            next_increments = next_increments[s_idx].to(self.DEVICE).to(torch.float32)


                                            # get rotation vectors
                                            R_marixes = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1), next_unclip_embs.squeeze(1))

                                            cos_sim = torch.cosine_similarity(take_base_unclip_embs, take_text_embs.to(self.DEVICE), dim = -1)
                                            next_text_hid_states = self.RV.cosin_rotate(take_text_hid_states, cos_sim.to(torch.float32).to(self.DEVICE), R_marixes,  power = self.pow_rote)

                                            # dif predict from base predict
                                            next_pred_unclip_embs = self.model(
                                                text_hidden_states = next_text_hid_states,
                                                prior_embeds = next_unclip_embs,
                                                rise = next_increments,
                                                )
                                            #print("\rnext_pred_unclip_embs", next_pred_unclip_embs.shape, end="")

                                            # combi_loss
                                            rote_loss, diff_loss = self.combi_loss(next_base_img_embs, next_image_embs,
                                                                                    next_unclip_embs, next_pred_unclip_embs)
                                            weight_regress_loss = 0.5
                                            if len(self.hist["acc"]):
                                                weight_regress_loss = np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])
                                                # weight of diff_loss depend from diff_step
                                                
                                            level = diff_step**0.5
                                            weight_regress_loss = weight_regress_loss ** level

                                            batch_size = len(s_idx)
                                            way_rote_loss += weight_regress_loss * rote_loss/batch_size
                                            way_diff_loss += weight_regress_loss * diff_loss/batch_size

                                            # control NAN INF in loss
                                            if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                                print(f"\rMovi {id_movi} rote_loss regress_step isnan or isinf, {type_ways[way]}_way")
                                                self.logs["naninf_rote"].append(id_movi)
                                                break

                                            # control NAN INF in loss
                                            if torch.isnan(way_diff_loss).sum() or torch.isinf(way_diff_loss).sum():
                                                print(f"\rMovi {id_movi} diff_loss regress_step isnan or isinf, {type_ways[way]}_way")
                                                self.logs["naninf_diff"].append(id_movi)
                                                break

                                            del(take_text_embs, take_base_unclip_embs)
                                            del(next_image_embs, next_base_img_embs, next_increments, next_text_hid_states)
                                            del(take_text_hid_states,  next_unclip_embs)
                                            #del(pred_unclip_embs, next_pred_unclip_embs)
                                            self.flush_memory()
                                            print(f"\rRegress_step {diff_step} with weight_regress_loss {weight_regress_loss:.3f}, way_rote_loss {way_rote_loss.mean().item()}, way_diff_loss {way_diff_loss.mean().item()}", end="")



                                # collect loss
                                way_loss = way_rote_loss + way_diff_loss # (0) for 1280
                                way_loss.backward(torch.ones_like(way_loss))
                                print(f"\rWay_loss.backward on {id_m} {type_ways[way]}_way", end="")

                                eph_loss_diff += way_diff_loss.mean().item()
                                eph_loss_rote += abs(way_rote_loss.mean().item())
                                eph_loss += way_loss.mean().item()


                                if clip_grad_norm:
                                    # make clip_grad_norm model
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                                self.optimizer.step()

                                # flush_memory
                                del(way_rote_loss, way_diff_loss)
                                del(pred_unclip_embs)
                                self.flush_memory()

                            else:

                              self.logs["bad_movi"].append(id_movi)
                              print(f'\rMovi {id_movi} has some static frames', end="")

                        # clear

                        del(increments_base, diff_img_embs)
                        self.flush_memory()

                    else:
                      print(f'Movi {id_movi} empty')
                      self.logs["empty"].append(id_movi)


                    del(image_embeds)
                    del(base_img_embs, img_embs)
                    del(text_hid_states, base_unclip_embs)

                # collect data
                good_steps = len(take_data)*ways
                eph_cos_acc/=good_steps
                eph_loss/=good_steps
                eph_loss_diff/=good_steps
                eph_loss_rote/=good_steps
                eph_base_loss/=good_steps

                self.hist["lr"].append(cur_lr)
                self.hist["loss"].append(eph_loss)
                self.hist["loss_diff"].append(eph_loss_diff)
                self.hist["loss_rote"].append(eph_loss_rote)
                self.hist["acc"].append(eph_cos_acc)
                self.hist["base_loss"].append(eph_base_loss)

                self.scheduler.step()

                 # compute av_weighted losses and acc by window
                if self.window:
                    av_acc =   np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])
                    av_diff =   np.mean(self.hist["loss_diff"][-min(len(self.hist["loss_diff"]), self.window):])
                    av_rote =  np.mean(self.hist["loss_rote"][-min(len(self.hist["loss_rote"]), self.window):])
                    av_loss =  np.mean(self.hist["loss"][-min(len(self.hist["loss"]), self.window):])

                else:
                    av_acc =  self.hist["acc"][-1]
                    av_diff =  self.hist["loss_diff"][-1]
                    av_rote =  self.hist["loss_rote"][-1]
                    av_loss =  self.hist["loss"][-1]

                if epoch - self.last_epoch > self.update_best:

                    if self.best_acc < av_acc:
                        self.best_acc = av_acc
                        self.last_save = "acc"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_diff > av_diff:
                        self.best_loss_diff = av_diff
                        self.last_save = "diff"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_rote > av_rote:
                        self.best_loss_rote = av_rote
                        self.last_save = "rote"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss > av_loss:
                        self.best_loss = av_loss
                        self.last_save = "loss"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    # same pereodicaly station model to check
                    if (epoch - self.last_epoch) and not epoch % self.fr_sv_cpt:
                        text += f'- save_{epoch}_ep_cpt '
                        model_name = f"/tmp_{epoch}_a_{av_acc:.3f}_l_{av_loss:.3f}_c_{av_rote:.3f}_m_{av_diff:.3f}.pt"
                        torch.save(self.model.state_dict(), self.dict_model_paths["temp"] + model_name)

                    if not save_model:
                        self.wait_train+=1
                        text += f'wait_{self.wait_train}'
                    else:
                      self.wait_train = 0 - self.window
                      self.wait2end = 0

                # finish training with loading best to predict
                if self.wait_train  > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1 or not epoch % self.fr_sv_cpt:

                    if self.wait_train > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1:
                        # load last best state
                        self.model.load_state_dict(torch.load(self.dict_model_paths[self.last_save]))
                        text += f' - load best_{self.last_save}_model {self.best_eph[self.last_save]} ep'

                    if self.wait_train > self.JDUN_TRAIN:
                        # update scheduler and optimizer
                        self.GAMMA *= self.EXP_RATE
                        self.LR_RATE =  self.hist["lr"][-1]*self.EXP_RATE
                        for g in self.optimizer.param_groups:
                            g['lr'] = self.LR_RATE
                        self.scheduler.gamma=self.GAMMA
                        self.scheduler.last_epoch = epoch
                        self.wait_train = 0
                        self.wait2end += 1


                    if self.wait2end > self.JDUN_END or epoch == self.last_epoch + self.EPOCHS -1 or not epoch % self.fr_sv_cpt:

                        if epoch and not epoch % self.fr_sv_cpt:
                            text += f' - save best_{self.last_save}_model {self.best_eph[self.last_save]} ep checkpoint'

                        if (epoch - self.last_epoch):
                            # take hist before best
                            self.hist = {key: self.hist[key][:self.best_eph[self.last_save]+1] for key in self.hist.keys()}
                            # update last history
                            update_dict_hist = {key: self.history_train[key] + self.hist[key] for key in self.history_train.keys()}

                            name_model = str(self.model_class) if not self.change_name_to_save else str(self.change_name_to_save)
                            # save checkpoint
                            torch.save({
                                'model_class' : self.model_class,
                                'saved_model' : self.last_save,
                                'epoch': self.best_eph[self.last_save],
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.best_loss,
                                'rote_loss': self.best_loss_rote,
                                'diff_loss': self.best_loss_diff,
                                'acc': self.best_acc,
                                'last_lr': self.hist["lr"][-1],
                                'last_checkpoint' : self.last_checkpoint,
                                'last_history': self.hist,
                                'all_history': update_dict_hist,
                                }, self.path_save + f'/{name_model}_{self.last_save}_{self.best_eph[self.last_save]}.cpt')

                        if (epoch - self.last_epoch) and self.wait2end > self.JDUN_END:
                            print(f"\nStop train, don't good fitness already on {epoch} ep, save best model from {self.best_eph[self.last_save]} ep")
                            break

                        if epoch == self.last_epoch + self.EPOCHS -1:
                            print(f'\nTrain is finished, saved the best model from {self.best_eph[self.last_save]}_ep')

                print(f'\rEp {epoch} all_loss {av_loss:.5f} | acc {av_acc:.5f} | diff_loss {av_diff:.5f} | rote_loss {av_rote:.5f} | lr {cur_lr:.5e} {text}\n')

        ###
        def flush_memory(self):
              import gc
              gc.collect()
              torch.cuda.empty_cache()
              torch.cuda.ipc_collect()
              with torch.no_grad():
                  for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()




class RoteTrainer():
        def __init__(self,
                      model,
                      device,
                      path_save: str,
                      next_train = False,
                      path_cpt = None,

                      ):
              super(RoteTrainer, self).__init__()
              self.path_save = path_save
              self.next_train = next_train
              self.DEVICE = device
              self.model = model
              self.emb = model.emb_dim
              self.model_class = self.model.__class__.__name__

              if self.next_train:
                  self.checkpoint = torch.load(path_cpt)
                  self.model.load_state_dict(self.checkpoint['model_state_dict'])
                  self.last_save = self.checkpoint['saved_model']
                  self.last_epoch = self.checkpoint['epoch']
                  self.last_lr = self.checkpoint["last_history"]["lr"][-1] #
                  self.best_loss = self.checkpoint['loss']
                  self.best_loss_mse = self.checkpoint['mse_loss']
                  self.best_loss_rote = self.checkpoint['rote_loss']
                  self.best_acc = self.checkpoint['acc']
                  self.history_train = self.checkpoint['all_history']
                  self.best_eph = {"loss" : self.last_epoch,
                                  "rote" : self.last_epoch,
                                  "mse" : self.last_epoch,
                                  "acc" : self.last_epoch,
                                  }
                  self.last_checkpoint =  str(path_cpt.split("/")[-1])

              # state for start
              else:
                  self.last_checkpoint = "New"
                  self.last_save = ''
                  self.last_epoch = 0
                  self.best_loss = 1000000
                  self.best_loss_mse = 1000000
                  self.best_loss_rote = 1000000
                  self.best_acc = 0
                  self.best_eph = {"loss" : 0,
                                  "rote" : 0,
                                  "mse" : 0,
                                  "acc" : 0,
                                  }

                  self.history_train = {"loss" : [],
                                        "loss_rote" : [],
                                        "loss_mse" : [],
                                        "acc" : [],
                                        "lr" : [],
                                        "base_loss" : []
                                        }

                  self.best_eph = {"loss" : 0,
                                    "rote" : 0,
                                    "mse" : 0,
                                    "acc" : 0
                                    }

              self.hist = {"loss" : [],
                          "loss_rote" : [],
                          "loss_mse" : [],
                          "acc" : [],
                          "base_loss" : [],
                          "lr" : []
                  }

              self.logs = {"bad_movi" : [],
                          "empty" : [],
                          "naninf_loss" : [],
                          "naninf_mse" : [],
                          "naninf_acc" : [],
                          "naninf_rote" : []
                  }

              # paths to save temory weights
              PATH_model_loss = self.path_save +'/last_best_loss.pt'
              PATH_model_rote = self.path_save +'/last_best_rote.pt'
              PATH_model_mse = self.path_save +'/last_best_mse.pt'
              PATH_model_acc = self.path_save +'/last_best_acc.pt'

              foder_temp_checkpoints = self.path_save + "/temp_checkpoints"
              if not os.path.exists(foder_temp_checkpoints):
                   os.makedirs(foder_temp_checkpoints)

              self.dict_model_paths = {"loss" : PATH_model_loss,
                                        "rote" : PATH_model_rote,
                                        "mse" : PATH_model_mse,
                                        "acc" : PATH_model_acc,
                                        "temp": foder_temp_checkpoints
                                        }

              # init waitings jdun
              self.wait_train = 0
              self.wait2end = 0

        def trailoop(self,
                     train_data,
                     max_batch : int,
                     optimizer: object,
                     scheduler: object,
                     maker_points: object,
                     rotator: object,
                     combine_loss:object,
                     cust_accuracy: object,
                     epochs: int,
                     friq_save_checkpoint = 15,
                     jdun_train = 3,
                     jdun_end = 3,
                     use_on_epoch = .9,
                     exp_rate = 0.99,
                     window = 3,
                     update_best = 0,
                     learning_rate = 1e-03,
                     update_lr = False,
                     reset_losses_acc = False,
                     suff_direct =  False,
                     add_back_train = False,
                     add_rote_train = False,
                     add_diff_train = False,
                     pow_rote = 1,
                     clip_grad_norm = False,
                     change_name_to_save = None,
                     ):

            # correct save, train and lerning rate
            self.suff_direct =  suff_direct
            self.add_back_train = add_back_train
            self.add_rote_train = add_rote_train
            self.add_diff_train = add_diff_train
            self.pow_rote = pow_rote
            self.fr_sv_cpt = friq_save_checkpoint
            self.window = window
            self.JDUN_END = jdun_end
            self.JDUN_TRAIN = jdun_train
            self.max_batch = max_batch
            self.use_on_epoch = use_on_epoch
            self.train_data = train_data
            if epochs < max(epochs, window+2):
                print(f"EPOCHS have to >= window+2, chahged onto {window+2}")
            self.EPOCHS = max(epochs, window+2)
            self.update_lr = update_lr
            self.LR_RATE = learning_rate
            self.EXP_RATE = exp_rate
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.GAMMA = self.scheduler.gamma
            self.reset_losses_acc = reset_losses_acc
            self.change_name_to_save = change_name_to_save

            # Custom loss and acc
            self.cos_acc_rote = cust_accuracy
            self.combi_loss = combine_loss

            # ComputeDiffPoints class
            self.maker_points = maker_points

            # inite rotation class
            self.RV = rotator()

            # wait to update_best
            self.update_best = update_best + self.window

            # ways in train
            ways = 2 if self.add_back_train else 1
            type_ways = ["normal", "back"]

            # correction for next_train
            if self.next_train:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.LR_RATE  if self.update_lr else self.last_lr
                self.scheduler.last_epoch = self.last_epoch

                if self.reset_losses_acc:
                      self.last_save = ''
                      self.last_epoch = 0
                      self.best_loss = 1000000
                      self.best_loss_mse = 1000000
                      self.best_loss_rote = 1000000
                      self.best_acc = 0
                      self.best_eph = {"loss" : 0,
                                      "rote" : 0,
                                      "mse" : 0,
                                      "acc" : 0,
                                      }



            for epoch  in tqdm(range(self.last_epoch, self.last_epoch + self.EPOCHS),
                               unit ="EPOHS", desc ="Пробегаемся по всем эпохам"):
                # inite
                self.flush_memory()
                text = ''
                save_model = 0
                eph_loss = 0
                eph_loss_rote = 0
                eph_loss_mse = 0
                eph_cos_acc = 0
                eph_base_loss = 0
                cos_acc = 0
                cur_lr = self.optimizer.param_groups[0]['lr']

                random.shuffle(self.train_data)  # shuffle  data each epoch
                take = int(len(self.train_data)*self.use_on_epoch)
                take_data = self.train_data[:take]  # take shuffle part data each epoch


                # go by selected data
                for  id_m, data  in tqdm(enumerate(take_data), unit = f" movi ",
                                                    desc ="Пробегаемся по всем фильмам"):
                    self.flush_memory()
                    # get next frame embbedings
                    id_movi, ids_frame = data['id_movi'], data['ids_frames']
                    text_hid_state = data['last_hidden_state']
                    text_embed =  data['text_embed']
                    unclip_embed = data['unclip_embed']
                    image_embeds = data['img_embeds']

                    # qty_frames used for normalization steps and control long movi too
                    qty_frames = len(image_embeds)

                    # control long movi
                    d_batch = self.max_batch-1
                    if qty_frames and id_movi:
                        if qty_frames <= self.max_batch:
                            d_batch = qty_frames -1

                        # take random ids in ids_frame[1:] because ids_frame[0] will first
                        rand_ids = [0] + list(np.random.choice(np.arange(1, qty_frames-1),
                                                                d_batch-1, replace=False))
                        rand_ids = sorted(rand_ids + [qty_frames - 1])  # Sort the selected indices

                        # select rand_ids labels and image_embeds
                        image_embeds = torch.concat([image_embeds[i] for i in rand_ids]).unsqueeze(dim=1)

                        # collect unclip tensors
                        base_unclip_embs = [unclip_embed for _ in range(d_batch)]
                        base_unclip_embs = torch.concat(base_unclip_embs).unsqueeze(dim=1)#

                        # collect text_hid_states tensors
                        text_hid_states = [text_hid_state for _ in range(d_batch)]
                        text_hid_states =  torch.concat(text_hid_states)

                        # collect text tensors
                        text_embs = [text_embed for _ in range(d_batch)]
                        text_embs =  torch.concat(text_embs).unsqueeze(dim=1)

                        # collect labels
                        labels = np.array([ids_frame[i] for i in rand_ids])

                        # place labels to class points
                        self.maker_points.time_labels = labels
                        config_norm, config_back = self.maker_points.getpoints_rotetrain()


                        for way in range(ways):

                            way_loss = 0
                            way_rote_loss = 0
                            way_mse_loss = 0
                            way_loss_base = 0

                            if not way: # normal way
                                all_points = self.maker_points.points

                            if way and self.add_back_train: # back way
                                all_points = (-1) * self.maker_points.back_points


                            # intit class for shufflee
                            b_srs = Shuff_Reshuff(len(all_points))
                            if self.suff_direct:
                                bs_idx = b_srs.shuffle()   # shuffleed indexees
                                bu_idx = b_srs.unshuffle() # indexees for re_shuffle
                            else:
                                bs_idx = b_srs.idx_base # shuffleed indexees
                                bu_idx = b_srs.idx_base # indexees for re_shuffle

                            ######### compute enters and states for model and losses for base ways

                            # collect time directions tensors and normalize
                            increments_base = torch.tensor(all_points/qty_frames).unsqueeze(1)

                            if not way: # normal way
                                # collect base_img_embs tensors
                                base_img_embs = [image_embeds[0] for _ in range(d_batch)]

                            if way and self.add_back_train: # back way
                                base_img_embs = [image_embeds[-1] for _ in range(d_batch)]

                            base_img_embs = torch.concat(base_img_embs).unsqueeze(dim=1)


                            if not way: # normal way
                                # collect img_embs tensors
                                img_embs = image_embeds[1:]

                            if way and self.add_back_train: # back way
                                # collect back_img_embs tensors
                                img_embs = torch.flip(image_embeds, [0,])[1:]

                            ####################################################################

                            # img difference
                            diff_img_embs =  (base_img_embs[bs_idx].squeeze(dim=1) - img_embs[bs_idx].squeeze(dim=1)).to(torch.float32) #

                            # check on static movi
                            bad_movi = 0
                            for k in range(diff_img_embs.shape[0]):
                              if abs(diff_img_embs[k].sum()) == 0:
                                bad_movi+=1

                            if not bad_movi:
                                # zero_grad optimizer
                                self.optimizer.zero_grad()

                                # base predict which can used in next step
                                pred_unclip_embs = self.model(
                                  text_hidden_states = text_hid_states[bs_idx].to(torch.float32).to(self.DEVICE), # shuffleed
                                  prior_embeds = base_unclip_embs[bs_idx].to(torch.float32).to(self.DEVICE),
                                  rise = increments_base[bs_idx].to(torch.float32).to(self.DEVICE)
                                                    )

                                # difference
                                rote_loss, mse_loss = self.combi_loss(base_img_embs[bs_idx], img_embs[bs_idx],
                                                                                base_unclip_embs[bs_idx], pred_unclip_embs)
                                way_rote_loss += rote_loss
                                way_mse_loss += mse_loss

                                # control NAN INF in loss
                                if torch.isnan(way_rote_loss).sum() or torch.isinf( way_rote_loss).sum():
                                  print(f"\rMovi {id_movi} rote_loss_base isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_rote"].append(id_movi)
                                  break


                                # control NAN INF in loss
                                if torch.isnan(way_mse_loss).sum() or torch.isinf(way_mse_loss).sum():
                                  print(f"\rMovi {id_movi} mse_loss_base isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_mse"].append(id_movi)
                                  break

                                # collect base_loss
                                way_loss_base += rote_loss + mse_loss # (0) for 1280
                                eph_base_loss += way_loss_base.mean().item()


                                cos_acc =  self.cos_acc_rote(init_img_vec = base_img_embs[bs_idx],
                                                             next_img_vec = img_embs[bs_idx],
                                                             init_unclip = base_unclip_embs[bs_idx],
                                                             pred_unclip = pred_unclip_embs,
                                                             )
                                                            # collect cos_acc
                                eph_cos_acc+= cos_acc.mean().item()

                                # control NAN INF in acc
                                if torch.isnan(cos_acc).sum() or torch.isinf(cos_acc).sum():
                                  print(f"\rMovi {id_movi} cos_acc_base isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_acc"].append(id_movi)
                                  break


                                # temp show losses for batch
                                if id_m:
                                  rate = id_m*ways
                                  print(f"\rMovi {id_movi} step {id_m} base_{type_ways[way]}_way mse {eph_loss_mse/rate:.5f} | rote {eph_loss_rote/rate:.5f} | lr {cur_lr:.5e}", end="")

                                #########  Rotation train steps ########################################
                                to_rote, rote_norm, rote_back = 0, 0, 0

                                if self.add_rote_train:#
                                    # Rotation train steps
                                    rote_norm = len(config_norm['id_uclip_emb'])

                                    if self.add_back_train:
                                        rote_back = len(config_back['id_uclip_emb'])

                                    # control size batch
                                    if not way:
                                      id_img_emb_s = np.array(config_norm['id_img_emb_s'])
                                      id_uclip_emb = np.array(config_norm['id_uclip_emb'])
                                      id_img_delta = np.array(config_norm['id_img_delta'])
                                      norm_delta = np.array(config_norm['norm_delta'])


                                    if way and rote_back:
                                      id_img_emb_s = np.array(config_back['id_img_emb_s'])
                                      id_uclip_emb = np.array(config_back['id_uclip_emb'])
                                      id_img_delta = np.array(config_back['id_img_delta'])
                                      norm_delta = (-1)*np.array(config_back['norm_delta'])

                                    to_rote =  rote_norm + rote_back

                                if self.add_rote_train and to_rote: #
                                    unique_id_img_emb = np.unique(id_img_emb_s)
                                    rote_steps = len(unique_id_img_emb)

                                    for unique_id in unique_id_img_emb:

                                        # take indes for eacc rote step
                                        set_dxs = np.where(id_img_emb_s == unique_id)[0]

                                        # intit class for shufflee again
                                        srs = Shuff_Reshuff(len(set_dxs))#[rote_norm, rote_back][way])
                                        if self.suff_direct:
                                            s_idx = srs.shuffle() # shuffleed indexees
                                        else:
                                            s_idx = srs.idx_base  # shuffleed indexees

                                        # collect norm steps
                                        take_text_hid_states = torch.clone(text_hid_states)[id_uclip_emb[set_dxs],:,:]
                                        take_base_unclip_embs = torch.clone(base_unclip_embs)[id_uclip_emb[set_dxs],:,:]
                                        take_base_img_embs = torch.clone(base_img_embs)[id_uclip_emb[set_dxs],:,:]
                                        take_text_embs = torch.clone(text_embs)[id_uclip_emb[set_dxs],:,:]
                                        base_img_embs_2rt = torch.clone(img_embs)[id_img_emb_s[set_dxs],:,:]
                                        image_embs_2rt = torch.clone(img_embs)[id_img_delta[set_dxs],:,:]
                                        increments_2rt = torch.tensor(norm_delta[set_dxs]).unsqueeze(1)

                                        # shufle
                                        take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_base_img_embs = take_base_img_embs[s_idx].to(self.DEVICE)
                                        take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                        base_img_embs_2rt = base_img_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                        image_embs_2rt = image_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                        increments_2rt = increments_2rt[s_idx].to(self.DEVICE)


                                        # get rotation marixes_i2i
                                        R_marixes_i2i = self.RV.get_rotation_matrix(take_base_img_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                    base_img_embs_2rt.squeeze(1).to(torch.float32)).to(self.DEVICE)

                                        # get cos_sim  base_img_embs vectors and base_unclip_embs vectors
                                        cos_sim_1 = torch.cosine_similarity(take_base_img_embs.to(self.DEVICE), take_base_unclip_embs, dim = -1)

                                        # compute roted unclip_embs with R_marixes_i2i and cos_sim base_img_embs and base_unclip_embs
                                        unclip_embs_2rt = self.RV.cosin_rotate(take_base_unclip_embs, cos_sim_1.to(self.DEVICE), R_marixes_i2i,  power = self.pow_rote)

                                        # get rotation marixes_u2u
                                        R_marixes_u2u = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                    unclip_embs_2rt.squeeze(1).to(torch.float32))

                                        cos_sim_2 = torch.cosine_similarity(take_base_unclip_embs.to(self.DEVICE), take_text_embs, dim = -1)
                                        text_hid_states_2rt = self.RV.cosin_rotate(take_text_hid_states, cos_sim_2.to(self.DEVICE), R_marixes_u2u, power = self.pow_rote)
                                        

                                        # rotation predict
                                        pred_rote_embs = self.model(
                                                      text_hidden_states = text_hid_states_2rt,
                                                      prior_embeds = unclip_embs_2rt,
                                                      rise = increments_2rt.to(torch.float32)
                                                                        )
                                        
                                        # combi_loss
                                        rote_loss, mse_loss = self.combi_loss(base_img_embs_2rt, image_embs_2rt,
                                                                              unclip_embs_2rt, pred_rote_embs)

                                        way_rote_loss += rote_loss/rote_steps
                                        way_mse_loss += mse_loss/rote_steps

                                        # control NAN INF in loss
                                        if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                          print(f"\rMovi {id_movi} rote_loss_rote isnan or isinf, {type_ways[way]}_way")
                                          self.logs["naninf_rote"].append(id_movi)
                                          break

                                        # control NAN INF in loss
                                        if torch.isnan(way_mse_loss).sum() or torch.isinf(way_mse_loss).sum():
                                          print(f"\rMovi {id_movi} mse_loss_rote isnan or isinf, {type_ways[way]}_way")
                                          self.logs["naninf_mse"].append(id_movi)
                                          break

                                        del(increments_2rt, take_text_embs, take_base_unclip_embs, take_base_img_embs, take_text_hid_states)
                                        del(pred_rote_embs, text_hid_states_2rt, R_marixes_i2i, R_marixes_u2u)
                                        del(unclip_embs_2rt, image_embs_2rt)
                                        self.flush_memory()

                                #########  Diff train steps ########################################
                                to_diff, diff_norm, diff_back = 0, 0, 0

                                # Diff train steps
                                if self.add_diff_train:
                                    config_diff_norm, config_diff_back = self.maker_points.getpoints_diftrain()
                                    diff_norm = len(config_diff_norm['id_uclip_emb'])

                                    if self.add_back_train:
                                        diff_back = len(config_diff_back['id_uclip_emb'])

                                    to_diff = diff_norm + diff_back

                                if self.add_diff_train and to_diff:

                                    # un_shuffleed
                                    pred_unclip_embs = torch.clone(pred_unclip_embs.detach().cpu())[bu_idx]

                                    if not way and diff_norm:
                                        # collect diff norm steps
                                        take_base_unclip_embs =torch.clone(base_unclip_embs)[config_diff_norm['id_uclip_emb'],:,:]
                                        take_text_hid_states = torch.clone(text_hid_states)[config_diff_norm['id_uclip_emb'],:,:]
                                        take_text_embs = torch.clone(text_embs)[config_diff_norm['id_uclip_emb'],:,:]
                                        next_unclip_embs = torch.clone(pred_unclip_embs)[config_diff_norm['id_uclip_emb'],:,:]
                                        next_base_img_embs = torch.clone(img_embs)[config_diff_norm['id_img_emb_s'],:,:]
                                        next_image_embs = torch.clone(img_embs)[config_diff_norm['id_img_delta'],:,:]
                                        next_increments = torch.tensor(config_diff_norm['norm_delta']).unsqueeze(1)


                                    if way and diff_back:
                                        # collect diff back steps
                                        take_base_unclip_embs = torch.clone(base_unclip_embs)[config_diff_back['id_uclip_emb'],:,:]
                                        take_text_hid_states = torch.clone(text_hid_states)[config_diff_back['id_uclip_emb'],:,:]
                                        take_text_embs = torch.clone(text_embs)[config_diff_back['id_uclip_emb'],:,:]
                                        next_unclip_embs = torch.clone(pred_unclip_embs)[config_diff_back['id_uclip_emb'],:,:]
                                        next_base_img_embs = torch.clone(img_embs)[config_diff_back['id_img_emb_s'],:,:]
                                        next_image_embs = torch.clone(img_embs)[config_diff_back['id_img_delta'],:,:]
                                        next_increments = (-1)*torch.tensor(config_diff_back['norm_delta']).unsqueeze(1)


                                    if (not way and diff_norm) or (way and diff_back):

                                        # intit class for shufflee again
                                        srs = Shuff_Reshuff([diff_norm, diff_back][way])
                                        if self.suff_direct:
                                            s_idx = srs.shuffle() # shuffleed indexees
                                        else:
                                            s_idx = srs.idx_base # shuffleed indexees

                                        # collect torch.concat
                                        take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                        next_unclip_embs = next_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        next_base_img_embs = next_base_img_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        next_image_embs = next_image_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        next_increments = next_increments[s_idx].to(self.DEVICE).to(torch.float32)


                                        # get rotation vectors
                                        R_marixes = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1), next_unclip_embs.squeeze(1))

                                        cos_sim = torch.cosine_similarity(take_base_unclip_embs, take_text_embs.to(self.DEVICE), dim = -1)
                                        next_text_hid_states = self.RV.cosin_rotate(take_text_hid_states, cos_sim.to(torch.float32).to(self.DEVICE), R_marixes,  power = self.pow_rote)

                                        # dif predict from base predict
                                        next_pred_unclip_embs = self.model(
                                            text_hidden_states = next_text_hid_states,
                                            prior_embeds = next_unclip_embs,
                                            rise = next_increments,
                                            )

                                        # combi_loss
                                        rote_loss, mse_loss = self.combi_loss(next_base_img_embs, next_image_embs,
                                                                                                next_unclip_embs, next_pred_unclip_embs)
                                        weight_diff_loss = 0.5
                                        if len(self.hist["acc"]):
                                            weight_diff_loss = np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])


                                        way_rote_loss += weight_diff_loss * rote_loss
                                        way_mse_loss += weight_diff_loss * mse_loss

                                        # control NAN INF in loss
                                        if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                            print(f"\rMovi {id_movi} rote_loss_diff isnan or isinf, {type_ways[way]}_way")
                                            self.logs["naninf_rote"].append(id_movi)
                                            break

                                        # control NAN INF in loss
                                        if torch.isnan(way_mse_loss).sum() or torch.isinf(way_mse_loss).sum():
                                            print(f"\rMovi {id_movi} mse_loss_diff isnan or isinf, {type_ways[way]}_way")
                                            self.logs["naninf_mse"].append(id_movi)
                                            break

                                        del(take_text_embs, take_base_unclip_embs)
                                        del(next_image_embs, next_base_img_embs, next_increments, next_text_hid_states)
                                        del(take_text_hid_states,  next_unclip_embs)
                                        del(pred_unclip_embs, next_pred_unclip_embs)
                                        self.flush_memory()
                                        print(f"\rWay_loss.backward on {id_m} step with weight_diff_loss {weight_diff_loss:.3f}, {type_ways[way]}_way", end="")


                                # collect loss
                                way_loss = way_rote_loss + way_mse_loss # (0) for 1280
                                way_loss.backward(torch.ones_like(way_loss))

                                eph_loss_mse += way_mse_loss.mean().item()
                                eph_loss_rote += abs(way_rote_loss.mean().item())
                                eph_loss += way_loss.mean().item()
                                

                                if clip_grad_norm:
                                    # make clip_grad_norm model
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                                self.optimizer.step()

                                # flush_memory
                                del(way_rote_loss, way_mse_loss)
                                self.flush_memory()

                            else:

                              self.logs["bad_movi"].append(id_movi)
                              print(f'\rMovi {id_movi} has some static frames', end="")

                        # clear
                        del(increments_base, diff_img_embs) 
                        self.flush_memory()

                    else:
                      print(f'Movi {id_movi} empty')
                      self.logs["empty"].append(id_movi)

                    del(image_embeds)
                    del(base_img_embs, img_embs)
                    del(text_hid_states, base_unclip_embs)

                # collect data
                good_steps = len(take_data)*ways
                eph_cos_acc/=good_steps
                eph_loss/=good_steps
                eph_loss_mse/=good_steps
                eph_loss_rote/=good_steps
                eph_base_loss/=good_steps

                self.hist["lr"].append(cur_lr)
                self.hist["loss"].append(eph_loss)
                self.hist["loss_mse"].append(eph_loss_mse)
                self.hist["loss_rote"].append(eph_loss_rote)
                self.hist["acc"].append(eph_cos_acc)
                self.hist["base_loss"].append(eph_base_loss)

                self.scheduler.step()

                 # compute av_weighted losses and acc by window
                if self.window:
                    av_acc =   np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])
                    av_mse =   np.mean(self.hist["loss_mse"][-min(len(self.hist["loss_mse"]), self.window):])
                    av_rote =  np.mean(self.hist["loss_rote"][-min(len(self.hist["loss_rote"]), self.window):])
                    av_loss =  np.mean(self.hist["loss"][-min(len(self.hist["loss"]), self.window):])

                else:
                    av_acc =  self.hist["acc"][-1]
                    av_mse =  self.hist["loss_mse"][-1]
                    av_rote =  self.hist["loss_rote"][-1]
                    av_loss =  self.hist["loss"][-1]

                if epoch - self.last_epoch > self.update_best:

                    if self.best_acc < av_acc:
                        self.best_acc = av_acc
                        self.last_save = "acc"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_mse > av_mse:
                        self.best_loss_mse = av_mse
                        self.last_save = "mse"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_rote > av_rote:
                        self.best_loss_rote = av_rote
                        self.last_save = "rote"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss > av_loss:
                        self.best_loss = av_loss
                        self.last_save = "loss"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    # same pereodicaly station model to check
                    if (epoch - self.last_epoch) and not epoch % self.fr_sv_cpt:
                        text += f'- save_{epoch}_ep_cpt '
                        model_name = f"/tmp_{epoch}_a_{av_acc:.3f}_l_{av_loss:.3f}_c_{av_rote:.3f}_m_{av_mse:.3f}.pt"
                        torch.save(self.model.state_dict(), self.dict_model_paths["temp"] + model_name)

                    if not save_model:
                        self.wait_train+=1
                        text += f'wait_{self.wait_train}'
                    else:
                      self.wait_train = 0 - self.window
                      self.wait2end = 0

                # finish training with loading best to predict
                if self.wait_train  > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1 or not epoch % self.fr_sv_cpt:

                    if self.wait_train > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1:
                        # load last best state
                        self.model.load_state_dict(torch.load(self.dict_model_paths[self.last_save]))
                        text += f' - load best_{self.last_save}_model {self.best_eph[self.last_save]} ep'

                    if self.wait_train > self.JDUN_TRAIN:
                        # update scheduler and optimizer
                        self.GAMMA *= self.EXP_RATE
                        self.LR_RATE =  self.hist["lr"][-1]*self.EXP_RATE
                        for g in self.optimizer.param_groups:
                            g['lr'] = self.LR_RATE
                        self.scheduler.gamma=self.GAMMA
                        self.scheduler.last_epoch = epoch
                        self.wait_train = 0
                        self.wait2end += 1


                    if self.wait2end > self.JDUN_END or epoch == self.last_epoch + self.EPOCHS -1 or not epoch % self.fr_sv_cpt:

                        if epoch and not epoch % self.fr_sv_cpt:
                            text += f' - save best_{self.last_save}_model {self.best_eph[self.last_save]} ep checkpoint'

                        if (epoch - self.last_epoch):
                            # take hist before best
                            self.hist = {key: self.hist[key][:self.best_eph[self.last_save]+1] for key in self.hist.keys()}
                            # update last history
                            update_dict_hist = {key: self.history_train[key] + self.hist[key] for key in self.history_train.keys()}

                            name_model = str(self.model_class) if not self.change_name_to_save else str(self.change_name_to_save)
                            # save checkpoint
                            torch.save({
                                'model_class' : self.model_class,
                                'saved_model' : self.last_save,
                                'epoch': self.best_eph[self.last_save],
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.best_loss,
                                'rote_loss': self.best_loss_rote,
                                'mse_loss': self.best_loss_mse,
                                'acc': self.best_acc,
                                'last_lr': self.hist["lr"][-1],
                                'last_checkpoint' : self.last_checkpoint,
                                'last_history': self.hist,
                                'all_history': update_dict_hist,
                                }, self.path_save + f'/{name_model}_{self.last_save}_{self.best_eph[self.last_save]}.cpt')

                        if (epoch - self.last_epoch) and self.wait2end > self.JDUN_END:
                            print(f"\nStop train, don't good fitness already on {epoch} ep, save best model from {self.best_eph[self.last_save]} ep")
                            break

                        if epoch == self.last_epoch + self.EPOCHS -1:
                            print(f'\nTrain is finished, saved the best model from {self.best_eph[self.last_save]}_ep')

                print(f'\rEp {epoch} all_loss {av_loss:.5f} | acc {av_acc:.5f} | mse_loss {av_mse:.5f} | rote_loss {av_rote:.5f} | lr {cur_lr:.5e} {text}\n')

        ###
        def flush_memory(self):
              import gc
              gc.collect()
              torch.cuda.empty_cache()
              torch.cuda.ipc_collect()
              with torch.no_grad():
                  for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


class Rote_W_Trainer():
        def __init__(self,
                      model,
                      device,
                      path_save: str,
                      next_train = False,
                      path_cpt = None,

                      ):
              super(Rote_W_Trainer, self).__init__()
              self.path_save = path_save
              self.next_train = next_train
              self.DEVICE = device
              self.model = model
              self.emb = model.emb_dim
              self.model_class = self.model.__class__.__name__

              if self.next_train:
                  self.checkpoint = torch.load(path_cpt)
                  self.model.load_state_dict(self.checkpoint['model_state_dict'])
                  self.last_save = self.checkpoint['saved_model']
                  self.last_epoch = self.checkpoint['epoch']
                  self.last_lr = self.checkpoint["last_history"]["lr"][-1] #
                  self.best_loss = self.checkpoint['loss']
                  self.best_loss_mse = self.checkpoint['mse_loss']
                  self.best_loss_rote = self.checkpoint['rote_loss']
                  self.best_acc = self.checkpoint['acc']
                  self.history_train = self.checkpoint['all_history']
                  self.best_eph = {"loss" : self.last_epoch,
                                  "rote" : self.last_epoch,
                                  "mse" : self.last_epoch,
                                  "acc" : self.last_epoch,
                                  }
                  self.last_checkpoint =  str(path_cpt.split("/")[-1])

              # state for start
              else:
                  self.last_checkpoint = "New"
                  self.last_save = ''
                  self.last_epoch = 0
                  self.best_loss = 1000000
                  self.best_loss_mse = 1000000
                  self.best_loss_rote = 1000000
                  self.best_acc = 0
                  self.best_eph = {"loss" : 0,
                                  "rote" : 0,
                                  "mse" : 0,
                                  "acc" : 0,
                                  }

                  self.history_train = {"loss" : [],
                                        "loss_rote" : [],
                                        "loss_mse" : [],
                                        "acc" : [],
                                        "lr" : [],
                                        "base_loss" : []
                                        }

                  self.best_eph = {"loss" : 0,
                                    "rote" : 0,
                                    "mse" : 0,
                                    "acc" : 0
                                    }

              self.hist = {"loss" : [],
                          "loss_rote" : [],
                          "loss_mse" : [],
                          "acc" : [],
                          "base_loss" : [],
                          "lr" : []
                  }

              self.logs = {"bad_movi" : [],
                          "empty" : [],
                          "naninf_loss" : [],
                          "naninf_mse" : [],
                          "naninf_acc" : [],
                          "naninf_rote" : []
                  }

              # paths to save temory weights
              PATH_model_loss = self.path_save +'/last_best_loss.pt'
              PATH_model_rote = self.path_save +'/last_best_rote.pt'
              PATH_model_mse = self.path_save +'/last_best_mse.pt'
              PATH_model_acc = self.path_save +'/last_best_acc.pt'

              foder_temp_checkpoints = self.path_save + "/temp_checkpoints"
              if not os.path.exists(foder_temp_checkpoints):
                   os.makedirs(foder_temp_checkpoints)

              self.dict_model_paths = {"loss" : PATH_model_loss,
                                        "rote" : PATH_model_rote,
                                        "mse" : PATH_model_mse,
                                        "acc" : PATH_model_acc,
                                        "temp": foder_temp_checkpoints
                                        }

              # init waitings jdun
              self.wait_train = 0
              self.wait2end = 0

        def trailoop(self,
                     train_data,
                     max_batch : int,
                     optimizer: object,
                     scheduler: object,
                     maker_points: object,
                     rotator: object,
                     combine_loss:object,
                     cust_accuracy: object,
                     epochs: int,
                     friq_save_checkpoint = 15,
                     jdun_train = 3,
                     jdun_end = 3,
                     use_on_epoch = .9,
                     exp_rate = 0.99,
                     window = 3,
                     update_best = 0,
                     learning_rate = 1e-03,
                     update_lr = False,
                     reset_losses_acc = False,
                     suff_direct =  False,
                     add_back_train = False,
                     add_rote_train = False,
                     add_diff_train = False,
                     pow_rote = 1,
                     clip_grad_norm = False,
                     change_name_to_save = None,
                     ):

            # correct save, train and lerning rate
            self.suff_direct =  suff_direct
            self.add_back_train = add_back_train
            self.add_rote_train = add_rote_train
            self.add_diff_train = add_diff_train
            self.pow_rote = pow_rote
            self.fr_sv_cpt = friq_save_checkpoint
            self.window = window
            self.JDUN_END = jdun_end
            self.JDUN_TRAIN = jdun_train
            self.max_batch = max_batch
            self.use_on_epoch = use_on_epoch
            self.train_data = train_data
            if epochs < max(epochs, window+2):
                print(f"EPOCHS have to >= window+2, chahged onto {window+2}")
            self.EPOCHS = max(epochs, window+2)
            self.update_lr = update_lr
            self.LR_RATE = learning_rate
            self.EXP_RATE = exp_rate
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.GAMMA = self.scheduler.gamma
            self.reset_losses_acc = reset_losses_acc
            self.change_name_to_save = change_name_to_save

            # Custom loss and acc
            self.cos_acc_rote = cust_accuracy
            self.combi_loss = combine_loss

            # ComputeDiffPoints class
            self.maker_points = maker_points

            # inite rotation class
            self.RV = rotator()

            # wait to update_best
            self.update_best = update_best + self.window

            # ways in train
            ways = 2 if self.add_back_train else 1
            type_ways = ["normal", "back"]

            # correction for next_train
            if self.next_train:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.LR_RATE  if self.update_lr else self.last_lr
                self.scheduler.last_epoch = self.last_epoch

                if self.reset_losses_acc:
                      self.last_save = ''
                      self.last_epoch = 0
                      self.best_loss = 1000000
                      self.best_loss_mse = 1000000
                      self.best_loss_rote = 1000000
                      self.best_acc = 0
                      self.best_eph = {"loss" : 0,
                                      "rote" : 0,
                                      "mse" : 0,
                                      "acc" : 0,
                                      }



            for epoch  in tqdm(range(self.last_epoch, self.last_epoch + self.EPOCHS),
                               unit ="EPOHS", desc ="Пробегаемся по всем эпохам"):
                # inite
                self.flush_memory()
                text = ''
                save_model = 0
                eph_loss = 0
                eph_loss_rote = 0
                eph_loss_mse = 0
                eph_cos_acc = 0
                eph_base_loss = 0
                cos_acc = 0
                cur_lr = self.optimizer.param_groups[0]['lr']

                random.shuffle(self.train_data)  # shuffle  data each epoch
                take = int(len(self.train_data)*self.use_on_epoch)
                take_data = self.train_data[:take]  # take shuffle part data each epoch


                # go by selected data
                for  id_m, data  in tqdm(enumerate(take_data), unit = f" movi ",
                                                    desc ="Пробегаемся по всем фильмам"):
                    self.flush_memory()
                    # get next frame embbedings
                    id_movi, ids_frame = data['id_movi'], data['ids_frames']
                    text_hid_state = data['last_hidden_state']
                    text_embed =  data['text_embed']
                    unclip_embed = data['unclip_embed']
                    image_embeds = data['img_embeds']

                    # qty_frames used for normalization steps and control long movi too
                    qty_frames = len(image_embeds)

                    # control long movi
                    d_batch = self.max_batch-1
                    if qty_frames and id_movi:
                        if qty_frames <= self.max_batch:
                            d_batch = qty_frames -1

                        # take random ids in ids_frame[1:] because ids_frame[0] will first
                        rand_ids = [0] + list(np.random.choice(np.arange(1, qty_frames-1),
                                                                d_batch-1, replace=False))
                        rand_ids = sorted(rand_ids + [qty_frames - 1])  # Sort the selected indices

                        # select rand_ids labels and image_embeds
                        image_embeds = torch.concat([image_embeds[i] for i in rand_ids]).unsqueeze(dim=1)

                        # collect unclip tensors
                        base_unclip_embs = [unclip_embed for _ in range(d_batch)]
                        base_unclip_embs = torch.concat(base_unclip_embs).unsqueeze(dim=1)#

                        # collect text_hid_states tensors
                        text_hid_states = [text_hid_state for _ in range(d_batch)]
                        text_hid_states =  torch.concat(text_hid_states)

                        # collect text tensors
                        text_embs = [text_embed for _ in range(d_batch)]
                        text_embs =  torch.concat(text_embs).unsqueeze(dim=1)

                        # collect labels
                        labels = np.array([ids_frame[i] for i in rand_ids])

                        # place labels to class points
                        self.maker_points.time_labels = labels
                        config_norm, config_back = self.maker_points.getpoints_rotetrain()


                        for way in range(ways):

                            way_loss = 0
                            way_rote_loss = 0
                            way_mse_loss = 0
                            way_loss_base = 0

                            if not way: # normal way
                                all_points = self.maker_points.points

                            if way and self.add_back_train: # back way
                                all_points = (-1) * self.maker_points.back_points


                            # intit class for shufflee
                            b_srs = Shuff_Reshuff(len(all_points))
                            if self.suff_direct:
                                bs_idx = b_srs.shuffle()   # shuffleed indexees
                                bu_idx = b_srs.unshuffle() # indexees for re_shuffle
                            else:
                                bs_idx = b_srs.idx_base # shuffleed indexees
                                bu_idx = b_srs.idx_base # indexees for re_shuffle

                            ######### compute enters and states for model and losses for base ways

                            # collect time directions tensors and normalize
                            increments_base = torch.tensor(all_points/qty_frames).unsqueeze(1)

                            if not way: # normal way
                                # collect base_img_embs tensors
                                base_img_embs = [image_embeds[0] for _ in range(d_batch)]

                            if way and self.add_back_train: # back way
                                base_img_embs = [image_embeds[-1] for _ in range(d_batch)]

                            base_img_embs = torch.concat(base_img_embs).unsqueeze(dim=1)


                            if not way: # normal way
                                # collect img_embs tensors
                                img_embs = image_embeds[1:]

                            if way and self.add_back_train: # back way
                                # collect back_img_embs tensors
                                img_embs = torch.flip(image_embeds, [0,])[1:]

                            ####################################################################

                            # img difference
                            diff_img_embs =  (base_img_embs[bs_idx].squeeze(dim=1) - img_embs[bs_idx].squeeze(dim=1)).to(torch.float32) #

                            # check on static movi
                            bad_movi = 0
                            for k in range(diff_img_embs.shape[0]):
                              if abs(diff_img_embs[k].sum()) == 0:
                                bad_movi+=1

                            if not bad_movi:
                                # zero_grad optimizer
                                self.optimizer.zero_grad()

                                # base predict which can used in next step
                                pred_unclip_embs = self.model(
                                  text_hidden_states = text_hid_states[bs_idx].to(torch.float32).to(self.DEVICE), # shuffleed
                                  prior_embeds = base_unclip_embs[bs_idx].to(torch.float32).to(self.DEVICE),
                                  rise = increments_base[bs_idx].to(torch.float32).to(self.DEVICE)
                                                    )

                                # difference
                                rote_loss, mse_loss = self.combi_loss(base_img_embs[bs_idx], img_embs[bs_idx],
                                                                                base_unclip_embs[bs_idx], pred_unclip_embs)
                                way_rote_loss += rote_loss
                                way_mse_loss += mse_loss

                                # control NAN INF in loss
                                if torch.isnan(way_rote_loss).sum() or torch.isinf( way_rote_loss).sum():
                                  print(f"\rMovi {id_movi} rote_loss_base isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_rote"].append(id_movi)
                                  break


                                # control NAN INF in loss
                                if torch.isnan(way_mse_loss).sum() or torch.isinf(way_mse_loss).sum():
                                  print(f"\rMovi {id_movi} mse_loss_base isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_mse"].append(id_movi)
                                  break

                                # collect base_loss
                                way_loss_base += rote_loss + mse_loss # (0) for 1280
                                eph_base_loss += way_loss_base.mean().item()


                                cos_acc =  self.cos_acc_rote(init_img_vec = base_img_embs[bs_idx],
                                                             next_img_vec = img_embs[bs_idx],
                                                             init_unclip = base_unclip_embs[bs_idx],
                                                             pred_unclip = pred_unclip_embs,
                                                             )
                                                            # collect cos_acc
                                eph_cos_acc+= cos_acc.mean().item()

                                # control NAN INF in acc
                                if torch.isnan(cos_acc).sum() or torch.isinf(cos_acc).sum():
                                  print(f"\rMovi {id_movi} cos_acc_base isnan or isinf, {type_ways[way]}_way")
                                  self.logs["naninf_acc"].append(id_movi)
                                  break


                                # temp show losses for batch
                                if id_m:
                                  rate = id_m*ways
                                  print(f"\rMovi {id_movi} step {id_m} base_{type_ways[way]}_way mse {eph_loss_mse/rate:.5f} | rote {eph_loss_rote/rate:.5f} | lr {cur_lr:.5e}", end="")

                                #########  Rotation train steps ########################################
                                to_rote, rote_norm, rote_back = 0, 0, 0

                                if self.add_rote_train:#
                                    # Rotation train steps
                                    rote_norm = len(config_norm['id_uclip_emb'])

                                    if self.add_back_train:
                                        rote_back = len(config_back['id_uclip_emb'])

                                    # control size batch
                                    if not way:
                                      id_img_emb_s = np.array(config_norm['id_img_emb_s'])
                                      id_uclip_emb = np.array(config_norm['id_uclip_emb'])
                                      id_img_delta = np.array(config_norm['id_img_delta'])
                                      norm_delta = np.array(config_norm['norm_delta'])


                                    if way and rote_back:
                                      id_img_emb_s = np.array(config_back['id_img_emb_s'])
                                      id_uclip_emb = np.array(config_back['id_uclip_emb'])
                                      id_img_delta = np.array(config_back['id_img_delta'])
                                      norm_delta = (-1)*np.array(config_back['norm_delta'])

                                    to_rote =  rote_norm + rote_back

                                if self.add_rote_train and to_rote: #
                                    unique_id_img_emb = np.unique(id_img_emb_s)
                                    rote_steps = len(unique_id_img_emb)

                                    steck_idx = []
                                    for unique_id in unique_id_img_emb:

                                        # take indes for eacc rote step
                                        set_dxs = np.where(id_img_emb_s == unique_id)[0]
                                        
                                        # collect steck
                                        steck_idx.append(set_dxs)

                                        # intit class for shufflee again
                                        srs = Shuff_Reshuff(len(set_dxs))#[rote_norm, rote_back][way])
                                        if self.suff_direct:
                                            s_idx = srs.shuffle() # shuffleed indexees
                                        else:
                                            s_idx = srs.idx_base  # shuffleed indexees

                                        # collect norm steps
                                        take_text_hid_states = torch.clone(text_hid_states)[id_uclip_emb[set_dxs],:,:]
                                        take_base_unclip_embs = torch.clone(base_unclip_embs)[id_uclip_emb[set_dxs],:,:]
                                        take_base_img_embs = torch.clone(base_img_embs)[id_uclip_emb[set_dxs],:,:]
                                        take_text_embs = torch.clone(text_embs)[id_uclip_emb[set_dxs],:,:]
                                        base_img_embs_2rt = torch.clone(img_embs)[id_img_emb_s[set_dxs],:,:]
                                        image_embs_2rt = torch.clone(img_embs)[id_img_delta[set_dxs],:,:]
                                        increments_2rt = torch.tensor(norm_delta[set_dxs]).unsqueeze(1)

                                        # shufle
                                        take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_base_img_embs = take_base_img_embs[s_idx].to(self.DEVICE)
                                        take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                        base_img_embs_2rt = base_img_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                        image_embs_2rt = image_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                        increments_2rt = increments_2rt[s_idx].to(self.DEVICE)


                                        # get rotation marixes_i2i
                                        R_marixes_i2i = self.RV.get_rotation_matrix(take_base_img_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                    base_img_embs_2rt.squeeze(1).to(torch.float32)).to(self.DEVICE)

                                        # get cos_sim  base_img_embs vectors and base_unclip_embs vectors
                                        cos_sim_1 = torch.cosine_similarity(take_base_img_embs.to(self.DEVICE), take_base_unclip_embs, dim = -1)

                                        # compute roted unclip_embs with R_marixes_i2i and cos_sim base_img_embs and base_unclip_embs
                                        unclip_embs_2rt = self.RV.cosin_rotate(take_base_unclip_embs, cos_sim_1.to(self.DEVICE), R_marixes_i2i,  power = self.pow_rote)

                                        # get rotation marixes_u2u
                                        R_marixes_u2u = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                    unclip_embs_2rt.squeeze(1).to(torch.float32))

                                        cos_sim_2 = torch.cosine_similarity(take_base_unclip_embs.to(self.DEVICE), take_text_embs, dim = -1)
                                        text_hid_states_2rt = self.RV.cosin_rotate(take_text_hid_states, cos_sim_2.to(self.DEVICE), R_marixes_u2u, power = self.pow_rote)
                                        

                                        # rotation predict
                                        pred_rote_embs = self.model(
                                                      text_hidden_states = text_hid_states_2rt,
                                                      prior_embeds = unclip_embs_2rt,
                                                      rise = increments_2rt.to(torch.float32)
                                                                        )
                                        
                                        # combi_loss
                                        rote_loss, mse_loss = self.combi_loss(base_img_embs_2rt, image_embs_2rt,
                                                                              unclip_embs_2rt, pred_rote_embs)

                                        way_rote_loss += rote_loss/rote_steps
                                        way_mse_loss += mse_loss/rote_steps

                                        # control NAN INF in loss
                                        if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                          print(f"\rMovi {id_movi} rote_loss_rote isnan or isinf, {type_ways[way]}_way")
                                          self.logs["naninf_rote"].append(id_movi)
                                          break

                                        # control NAN INF in loss
                                        if torch.isnan(way_mse_loss).sum() or torch.isinf(way_mse_loss).sum():
                                          print(f"\rMovi {id_movi} mse_loss_rote isnan or isinf, {type_ways[way]}_way")
                                          self.logs["naninf_mse"].append(id_movi)
                                          break

                                        del(increments_2rt, take_text_embs, take_base_unclip_embs, take_base_img_embs, take_text_hid_states)
                                        del(pred_rote_embs, text_hid_states_2rt, R_marixes_i2i, R_marixes_u2u)
                                        del(unclip_embs_2rt, image_embs_2rt)
                                        self.flush_memory()


                                    # make new steck id in vertical formate for new rote predictions
                                    new_steck_idx = [[] for _ in range(len(unique_id_img_emb))]
                                    for i in range(len(steck_idx)):
                                      for idxs in steck_idx:
                                        if i in range(len(idxs)):
                                          new_steck_idx[i].append(idxs[i])

                                    for set_dxs in new_steck_idx:

                                        # intit class for shufflee again
                                        srs = Shuff_Reshuff(len(set_dxs))#[rote_norm, rote_back][way])
                                        if self.suff_direct:
                                            s_idx = srs.shuffle() # shuffleed indexees
                                        else:
                                            s_idx = srs.idx_base  # shuffleed indexees

                                        # collect norm steps
                                        take_text_hid_states = torch.clone(text_hid_states)[id_uclip_emb[set_dxs],:,:]
                                        take_base_unclip_embs = torch.clone(base_unclip_embs)[id_uclip_emb[set_dxs],:,:]
                                        take_base_img_embs = torch.clone(base_img_embs)[id_uclip_emb[set_dxs],:,:]
                                        take_text_embs = torch.clone(text_embs)[id_uclip_emb[set_dxs],:,:]
                                        base_img_embs_2rt = torch.clone(img_embs)[id_img_emb_s[set_dxs],:,:]
                                        image_embs_2rt = torch.clone(img_embs)[id_img_delta[set_dxs],:,:]
                                        increments_2rt = torch.tensor(norm_delta[set_dxs]).unsqueeze(1)

                                        # shufle
                                        take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_base_img_embs = take_base_img_embs[s_idx].to(self.DEVICE)
                                        take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                        base_img_embs_2rt = base_img_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                        image_embs_2rt = image_embs_2rt[s_idx].to(self.DEVICE).to(torch.float32)
                                        increments_2rt = increments_2rt[s_idx].to(self.DEVICE)


                                        # get rotation marixes_i2i
                                        R_marixes_i2i = self.RV.get_rotation_matrix(take_base_img_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                    base_img_embs_2rt.squeeze(1).to(torch.float32)).to(self.DEVICE)

                                        # get cos_sim  base_img_embs vectors and base_unclip_embs vectors
                                        cos_sim_1 = torch.cosine_similarity(take_base_img_embs.to(self.DEVICE), take_base_unclip_embs, dim = -1)

                                        # compute roted unclip_embs with R_marixes_i2i and cos_sim base_img_embs and base_unclip_embs
                                        unclip_embs_2rt = self.RV.cosin_rotate(take_base_unclip_embs, cos_sim_1.to(self.DEVICE), R_marixes_i2i,  power = self.pow_rote)

                                        # get rotation marixes_u2u
                                        R_marixes_u2u = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1).to(torch.float32).to(self.DEVICE),
                                                                                    unclip_embs_2rt.squeeze(1).to(torch.float32))

                                        cos_sim_2 = torch.cosine_similarity(take_base_unclip_embs.to(self.DEVICE), take_text_embs, dim = -1)
                                        text_hid_states_2rt = self.RV.cosin_rotate(take_text_hid_states, cos_sim_2.to(self.DEVICE), R_marixes_u2u, power = self.pow_rote)


                                        # rotation predict
                                        pred_rote_embs = self.model(
                                                      text_hidden_states = text_hid_states_2rt,
                                                      prior_embeds = unclip_embs_2rt,
                                                      rise = increments_2rt.to(torch.float32)
                                                                        )

                                        # combi_loss
                                        rote_loss, mse_loss = self.combi_loss(base_img_embs_2rt, image_embs_2rt,
                                                                              unclip_embs_2rt, pred_rote_embs)

                                        way_rote_loss += rote_loss/rote_steps
                                        way_mse_loss += mse_loss/rote_steps

                                        print(f"\rvertical_way_rote_loss {way_rote_loss.mean().item()}, way_mse_loss {way_mse_loss.mean().item()}", end="")

                                        # control NAN INF in loss
                                        if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                          print(f"\rMovi {id_movi} rote_loss_rote isnan or isinf, {type_ways[way]}_way")
                                          self.logs["naninf_rote"].append(id_movi)
                                          break

                                        # control NAN INF in loss
                                        if torch.isnan(way_mse_loss).sum() or torch.isinf(way_mse_loss).sum():
                                          print(f"\rMovi {id_movi} mse_loss_rote isnan or isinf, {type_ways[way]}_way")
                                          self.logs["naninf_mse"].append(id_movi)
                                          break

                                        del(increments_2rt, take_text_embs, take_base_unclip_embs, take_base_img_embs, take_text_hid_states)
                                        del(pred_rote_embs, text_hid_states_2rt, R_marixes_i2i, R_marixes_u2u)
                                        del(unclip_embs_2rt, image_embs_2rt)
                                        self.flush_memory()




                                #########  Diff train steps ########################################
                                to_diff, diff_norm, diff_back = 0, 0, 0

                                # Diff train steps
                                if self.add_diff_train:
                                    config_diff_norm, config_diff_back = self.maker_points.getpoints_diftrain()
                                    diff_norm = len(config_diff_norm['id_uclip_emb'])

                                    if self.add_back_train:
                                        diff_back = len(config_diff_back['id_uclip_emb'])

                                    to_diff = diff_norm + diff_back

                                if self.add_diff_train and to_diff:

                                    # un_shuffleed
                                    pred_unclip_embs = torch.clone(pred_unclip_embs.detach().cpu())[bu_idx]

                                    if not way and diff_norm:
                                        # collect diff norm steps
                                        take_base_unclip_embs =torch.clone(base_unclip_embs)[config_diff_norm['id_uclip_emb'],:,:]
                                        take_text_hid_states = torch.clone(text_hid_states)[config_diff_norm['id_uclip_emb'],:,:]
                                        take_text_embs = torch.clone(text_embs)[config_diff_norm['id_uclip_emb'],:,:]
                                        next_unclip_embs = torch.clone(pred_unclip_embs)[config_diff_norm['id_uclip_emb'],:,:]
                                        next_base_img_embs = torch.clone(img_embs)[config_diff_norm['id_img_emb_s'],:,:]
                                        next_image_embs = torch.clone(img_embs)[config_diff_norm['id_img_delta'],:,:]
                                        next_increments = torch.tensor(config_diff_norm['norm_delta']).unsqueeze(1)


                                    if way and diff_back:
                                        # collect diff back steps
                                        take_base_unclip_embs = torch.clone(base_unclip_embs)[config_diff_back['id_uclip_emb'],:,:]
                                        take_text_hid_states = torch.clone(text_hid_states)[config_diff_back['id_uclip_emb'],:,:]
                                        take_text_embs = torch.clone(text_embs)[config_diff_back['id_uclip_emb'],:,:]
                                        next_unclip_embs = torch.clone(pred_unclip_embs)[config_diff_back['id_uclip_emb'],:,:]
                                        next_base_img_embs = torch.clone(img_embs)[config_diff_back['id_img_emb_s'],:,:]
                                        next_image_embs = torch.clone(img_embs)[config_diff_back['id_img_delta'],:,:]
                                        next_increments = (-1)*torch.tensor(config_diff_back['norm_delta']).unsqueeze(1)


                                    if (not way and diff_norm) or (way and diff_back):

                                        # intit class for shufflee again
                                        srs = Shuff_Reshuff([diff_norm, diff_back][way])
                                        if self.suff_direct:
                                            s_idx = srs.shuffle() # shuffleed indexees
                                        else:
                                            s_idx = srs.idx_base # shuffleed indexees

                                        # collect torch.concat
                                        take_base_unclip_embs = take_base_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_text_hid_states = take_text_hid_states[s_idx].to(self.DEVICE).to(torch.float32)
                                        take_text_embs = take_text_embs[s_idx].to(self.DEVICE)
                                        next_unclip_embs = next_unclip_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        next_base_img_embs = next_base_img_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        next_image_embs = next_image_embs[s_idx].to(self.DEVICE).to(torch.float32)
                                        next_increments = next_increments[s_idx].to(self.DEVICE).to(torch.float32)


                                        # get rotation vectors
                                        R_marixes = self.RV.get_rotation_matrix(take_base_unclip_embs.squeeze(1), next_unclip_embs.squeeze(1))

                                        cos_sim = torch.cosine_similarity(take_base_unclip_embs, take_text_embs.to(self.DEVICE), dim = -1)
                                        next_text_hid_states = self.RV.cosin_rotate(take_text_hid_states, cos_sim.to(torch.float32).to(self.DEVICE), R_marixes,  power = self.pow_rote)

                                        # dif predict from base predict
                                        next_pred_unclip_embs = self.model(
                                            text_hidden_states = next_text_hid_states,
                                            prior_embeds = next_unclip_embs,
                                            rise = next_increments,
                                            )

                                        # combi_loss
                                        rote_loss, mse_loss = self.combi_loss(next_base_img_embs, next_image_embs,
                                                                                                next_unclip_embs, next_pred_unclip_embs)
                                        weight_diff_loss = 0.5
                                        if len(self.hist["acc"]):
                                            weight_diff_loss = np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])


                                        way_rote_loss += weight_diff_loss * rote_loss
                                        way_mse_loss += weight_diff_loss * mse_loss

                                        # control NAN INF in loss
                                        if torch.isnan(way_rote_loss).sum() or torch.isinf(way_rote_loss).sum():
                                            print(f"\rMovi {id_movi} rote_loss_diff isnan or isinf, {type_ways[way]}_way")
                                            self.logs["naninf_rote"].append(id_movi)
                                            break

                                        # control NAN INF in loss
                                        if torch.isnan(way_mse_loss).sum() or torch.isinf(way_mse_loss).sum():
                                            print(f"\rMovi {id_movi} mse_loss_diff isnan or isinf, {type_ways[way]}_way")
                                            self.logs["naninf_mse"].append(id_movi)
                                            break

                                        del(take_text_embs, take_base_unclip_embs)
                                        del(next_image_embs, next_base_img_embs, next_increments, next_text_hid_states)
                                        del(take_text_hid_states,  next_unclip_embs)
                                        del(pred_unclip_embs, next_pred_unclip_embs)
                                        self.flush_memory()
                                        print(f"\rWay_loss.backward on {id_m} step with weight_diff_loss {weight_diff_loss:.3f}, {type_ways[way]}_way", end="")


                                # collect loss
                                way_loss = way_rote_loss + way_mse_loss # (0) for 1280
                                way_loss.backward(torch.ones_like(way_loss))

                                eph_loss_mse += way_mse_loss.mean().item()
                                eph_loss_rote += abs(way_rote_loss.mean().item())
                                eph_loss += way_loss.mean().item()
                                

                                if clip_grad_norm:
                                    # make clip_grad_norm model
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                                self.optimizer.step()

                                # flush_memory
                                del(way_rote_loss, way_mse_loss)
                                self.flush_memory()

                            else:

                              self.logs["bad_movi"].append(id_movi)
                              print(f'\rMovi {id_movi} has some static frames', end="")

                        # clear
                        del(increments_base, diff_img_embs) 
                        self.flush_memory()

                    else:
                      print(f'Movi {id_movi} empty')
                      self.logs["empty"].append(id_movi)

                    del(image_embeds)
                    del(base_img_embs, img_embs)
                    del(text_hid_states, base_unclip_embs)

                # collect data
                good_steps = len(take_data)*ways
                eph_cos_acc/=good_steps
                eph_loss/=good_steps
                eph_loss_mse/=good_steps
                eph_loss_rote/=good_steps
                eph_base_loss/=good_steps

                self.hist["lr"].append(cur_lr)
                self.hist["loss"].append(eph_loss)
                self.hist["loss_mse"].append(eph_loss_mse)
                self.hist["loss_rote"].append(eph_loss_rote)
                self.hist["acc"].append(eph_cos_acc)
                self.hist["base_loss"].append(eph_base_loss)

                self.scheduler.step()

                 # compute av_weighted losses and acc by window
                if self.window:
                    av_acc =   np.mean(self.hist["acc"][-min(len(self.hist["acc"]), self.window):])
                    av_mse =   np.mean(self.hist["loss_mse"][-min(len(self.hist["loss_mse"]), self.window):])
                    av_rote =  np.mean(self.hist["loss_rote"][-min(len(self.hist["loss_rote"]), self.window):])
                    av_loss =  np.mean(self.hist["loss"][-min(len(self.hist["loss"]), self.window):])

                else:
                    av_acc =  self.hist["acc"][-1]
                    av_mse =  self.hist["loss_mse"][-1]
                    av_rote =  self.hist["loss_rote"][-1]
                    av_loss =  self.hist["loss"][-1]

                if epoch - self.last_epoch > self.update_best:

                    if self.best_acc < av_acc:
                        self.best_acc = av_acc
                        self.last_save = "acc"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_mse > av_mse:
                        self.best_loss_mse = av_mse
                        self.last_save = "mse"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss_rote > av_rote:
                        self.best_loss_rote = av_rote
                        self.last_save = "rote"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    if self.best_loss > av_loss:
                        self.best_loss = av_loss
                        self.last_save = "loss"
                        text += f'- save_{self.last_save} '
                        self.best_eph[self.last_save] = epoch
                        torch.save(self.model.state_dict(), self.dict_model_paths[self.last_save])
                        save_model += 1

                    # same pereodicaly station model to check
                    if (epoch - self.last_epoch) and not epoch % self.fr_sv_cpt:
                        text += f'- save_{epoch}_ep_cpt '
                        model_name = f"/tmp_{epoch}_a_{av_acc:.3f}_l_{av_loss:.3f}_c_{av_rote:.3f}_m_{av_mse:.3f}.pt"
                        torch.save(self.model.state_dict(), self.dict_model_paths["temp"] + model_name)

                    if not save_model:
                        self.wait_train+=1
                        text += f'wait_{self.wait_train}'
                    else:
                      self.wait_train = 0 - self.window
                      self.wait2end = 0

                # finish training with loading best to predict
                if self.wait_train  > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1 or not epoch % self.fr_sv_cpt:

                    if self.wait_train > self.JDUN_TRAIN or epoch == self.last_epoch + self.EPOCHS-1:
                        # load last best state
                        self.model.load_state_dict(torch.load(self.dict_model_paths[self.last_save]))
                        text += f' - load best_{self.last_save}_model {self.best_eph[self.last_save]} ep'

                    if self.wait_train > self.JDUN_TRAIN:
                        # update scheduler and optimizer
                        self.GAMMA *= self.EXP_RATE
                        self.LR_RATE =  self.hist["lr"][-1]*self.EXP_RATE
                        for g in self.optimizer.param_groups:
                            g['lr'] = self.LR_RATE
                        self.scheduler.gamma=self.GAMMA
                        self.scheduler.last_epoch = epoch
                        self.wait_train = 0
                        self.wait2end += 1


                    if self.wait2end > self.JDUN_END or epoch == self.last_epoch + self.EPOCHS -1 or not epoch % self.fr_sv_cpt:

                        if epoch and not epoch % self.fr_sv_cpt:
                            text += f' - save best_{self.last_save}_model {self.best_eph[self.last_save]} ep checkpoint'

                        if (epoch - self.last_epoch):
                            # take hist before best
                            self.hist = {key: self.hist[key][:self.best_eph[self.last_save]+1] for key in self.hist.keys()}
                            # update last history
                            update_dict_hist = {key: self.history_train[key] + self.hist[key] for key in self.history_train.keys()}

                            name_model = str(self.model_class) if not self.change_name_to_save else str(self.change_name_to_save)
                            # save checkpoint
                            torch.save({
                                'model_class' : self.model_class,
                                'saved_model' : self.last_save,
                                'epoch': self.best_eph[self.last_save],
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.best_loss,
                                'rote_loss': self.best_loss_rote,
                                'mse_loss': self.best_loss_mse,
                                'acc': self.best_acc,
                                'last_lr': self.hist["lr"][-1],
                                'last_checkpoint' : self.last_checkpoint,
                                'last_history': self.hist,
                                'all_history': update_dict_hist,
                                }, self.path_save + f'/{name_model}_{self.last_save}_{self.best_eph[self.last_save]}.cpt')

                        if (epoch - self.last_epoch) and self.wait2end > self.JDUN_END:
                            print(f"\nStop train, don't good fitness already on {epoch} ep, save best model from {self.best_eph[self.last_save]} ep")
                            break

                        if epoch == self.last_epoch + self.EPOCHS -1:
                            print(f'\nTrain is finished, saved the best model from {self.best_eph[self.last_save]}_ep')

                print(f'\rEp {epoch} all_loss {av_loss:.5f} | acc {av_acc:.5f} | mse_loss {av_mse:.5f} | rote_loss {av_rote:.5f} | lr {cur_lr:.5e} {text}\n')

        ###
        def flush_memory(self):
              import gc
              gc.collect()
              torch.cuda.empty_cache()
              torch.cuda.ipc_collect()
              with torch.no_grad():
                  for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()                   
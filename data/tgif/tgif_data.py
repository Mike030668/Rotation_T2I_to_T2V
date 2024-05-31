import requests
import numpy as np
import pandas as pd
import os
import PIL


class TakeTgif():

    """
    make before !git clone https://github.com/raingo/TGIF-Release.git -q

    tsv_path - path to tgif-v1.0.tsv file or similar
    txt_path - path to train.txt file or similar
    """

    def __init__(self,
                 tsv_path: str,
                 txt_path: str
                 ):
        
        super(TakeTgif, self).__init__()
        
        self.gif_df = pd.read_csv(tsv_path, sep='\t', header=None)
        
        with open(txt_path) as f:
            self.train_vids = [l.strip() for l in f.readlines()]  # urls
            self.train_df = self.gif_df[self.gif_df[0].isin(self.train_vids)]
            self.train_vidxs, self.train_corpus = list(self.train_df.index.values), list(self.train_df[1])

    def selector(self,
                 qty = None,
                 patterns_to_search = 'random', 
                 pass_lst = [],                    
        ):
        """

        
        """
        self.patterns_to_search = patterns_to_search
      
        print("self.train_df ", self.train_df.shape)  
        select_df = self.train_df[~self.train_df.index.isin(pass_lst)] if pass_lst else self.train_df
        print("select_df ", select_df.shape)  

        if self.patterns_to_search == 'random':
            
            self.rand_gif_df = select_df.sample(qty) if qty else select_df
            print("rand_gif_df ", self.rand_gif_df.shape) 


            self.train_vidxs, self.train_corpus = list(self.rand_gif_df.index.values), list(self.rand_gif_df[1])
            self.matchs = [(id, text) for id, text in zip(self.train_vidxs, self.train_corpus)]

        elif type(self.patterns_to_search) == list :
            self.matchs = [(self.train_vidxs[i], s) for (i, s) in enumerate(self.train_corpus) if all([(p in s) for p in self.patterns_to_search])]

    def seacher(
            self,
            min_size : int,
            take_min : int,
            wait_len : int,
            wait_movis : int,
            file_path : str,
            path_tmp = '/content',
            save_each = 10,
            dir_image = '',
            ):

            """
            selector movies from tgif_dataset
            https://github.com/raingo/TGIF-Release/tree/master

            path_gif -  save path tempory
            
            """


            # for save information to dataframe

            self._min_size = min_size
            self._take_min = take_min
            self.take_max = wait_len + 1

            path_gif = path_tmp + '/temp.gif' # save path tempory
            
            #if len(dir_image):
            #   dir_name = dir_image + '/images'   # save path tempory
            #    try: os.mkdir(dir_name)
            #    except: pass

            select_movis = 0
            pil_df = []
            dif_frames = np.array([])

            for id_matchs in range(len(self.matchs)):
                bad_movi = 0
                low_movi = 0
                error = 0
                id_movi = self.matchs[id_matchs][0]
                caption = self.rand_gif_df.loc[id_movi][1]
                url =  self.rand_gif_df.loc[id_movi][0]

                try:
                    #download gif
                    with open(path_gif, 'wb') as f:
                        f.write(requests.get(url).content)

                    # open gif
                    gif_img = PIL.Image.open(path_gif)

                    # take original frames
                    frames = [frame.convert('RGB') for frame in PIL.ImageSequence.Iterator(gif_img)]
                    width, height = frames[0].size
                except:
                    print(f"\rNon or bad gif", end="")
                    error = 1
                    frames = []
                    width, height = 0, 0

                if len(frames) < self._take_min:
                    if error == 1: error = 0
                    else: print(f"\rSmall gif {id_movi} with {len(frames)} frames", end="")

                # select movi not shot than take_min
                else:
                    # select movi not low than MIN_SIZE
                    if width < self._min_size  or  height < self._min_size:
                        print(f'\rMovi {id_movi} have low H_{height} or W_{width}', end="")
                        low_movi = 1

                    if len(frames) > self.take_max and not low_movi:
                        # chack some statick movi
                        dif_frames = [(frame - np.array(frames[0])) for frame in frames[1:]]
                        dif_frames = np.array(dif_frames)

                        for k in range(dif_frames.shape[0]):
                            if dif_frames[k].sum() == 0:
                                bad_movi+=1

                        if not bad_movi:
                            frames = frames[:self.take_max]
                            print(f'\rCut movi {id_movi}', end="")

                    else:
                        if bad_movi:
                            print(f"\rMovi {id_movi} have static frames\n")


                    if not bad_movi and not low_movi:
                        # save frames and make
                        for idx in range(len(frames)):
                            img_path = os.path.join(dir_image, f'{id_movi}_{idx}.jpg')
                            
                            if len(dir_image):
                               frames[idx].save(img_path)

                            pil_df.append([id_movi, f"{width},{height}", idx, [width, height], caption, img_path, self.patterns_to_search, url])

                        select_movis += 1
                        print(f"\rMovi {id_movi} have len = {len(frames)} take all move with W_{width} x H_{height} selected_{select_movis}\n")

                    if select_movis >= wait_movis: break

                if len(pil_df) and not len(pil_df) % save_each:
                    df = pd.DataFrame(pil_df)
                    df.columns = ['id_movi', "w x h", 'id_frame', 'size_frame', 'caption_movi', 'paths', 'patterns' , 'url']
                    df.to_csv(file_path, index=False)

            df = pd.DataFrame(pil_df)
            df.columns = ['id_movi', "w x h", 'id_frame', 'size_frame', 'caption_movi', 'paths', 'patterns' , 'url']
            df.to_csv(file_path, index=False)


    def load_movies(self,
                    df_path: pd.DataFrame,
                    path_tmp = '/content',
                
        ):

        self.df = pd.read_csv(df_path)
        id_movies = self.df["id_movi"].unique()

        path_gif = path_tmp + '/temp.gif' # save path tempory

        dir_name = path_tmp + '/images'   # save path tempory
        try: os.mkdir(dir_name)
        except: pass
        
        saved_movis = 1
        for id_movi in id_movies:
            url =  self.train_df.loc[id_movi][0]
            len_movi = self.df[self.df["id_movi"] == id_movi].shape[0]
            #download gif
            with open(path_gif, 'wb') as f:
                f.write(requests.get(url).content)

            # open gif
            gif_img = PIL.Image.open(path_gif)
            # take original frames
            frames = [frame.convert('RGB') for frame in PIL.ImageSequence.Iterator(gif_img)][:len_movi]
            
            
            # save frames and make
            for id_frame in range(len_movi):
                img_path = os.path.join(dir_name, f'{id_movi}_{id_frame}.jpg')
                frames[id_frame].save(img_path)

            print(f"\rMovi {id_movi} have len = {len_movi} saved_{saved_movis}\n")
            saved_movis+=1
import glob
import PIL
import base64
from IPython import display


import cv2
import numpy as np
# Функции для возпроизведение видео с результатом
from moviepy.editor import *


def make_gif(set_dir_pil,
             out_name = "my_awesome.gif",
             duration = 10,
             loop=0,
             ext = '/*.JPG'):

    if type(set_dir_pil) == str:
       frames = [PIL.Image.open(image) for image in glob.glob(f"{set_dir_pil}{ext}")]

    elif type(set_dir_pil) == list: frames = set_dir_pil

    frame_one = frames[0]
    frame_one.save(out_name, format="GIF", append_images=frames,
               save_all=True, optimize=False, duration=duration, loop=loop)


def show_gif(fname):
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')
    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')




def make_mp4(pil_imgs, out_name, width, height, duration=20):
  # Чем больше кадров в секунду, тем быстее видео будет проигрываться
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'MP4V'), duration, (width, height))

    # В цикле добавляем каждый кадр в видео (делаем предобработку кадра - меняем каналы с RGB в BGR
    # это нужно потому что cv2 воспринимает каналы как BGR)
    for frame in pil_imgs:
      out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

    # Закрываем объект для создания видео
    out.release()


def show_mp4(path, width, duration=20):
    # Извлекаем видео из заданного пути (куда мы ранее записыли видео через cv2)
    clip=VideoFileClip(path)

    # Отображаем видео в колабе
    return clip.ipython_display(width=width, maxduration = duration)
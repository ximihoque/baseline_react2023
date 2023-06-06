import os
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import random
import sys
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

# ---------- CONFIG ------------
IMG_SIZE = 256
CROP_SIZE = 224
CLIP_LENGTH = 751
# ------------------------------

class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = torch.nn.Sequential(
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            normalize
        )
        img = transform(img)
        return img

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def extract_video_features(video_path, img_transform, clip_length):
    video_list = []
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
#     print ('FPS: ', fps)
#     print ('#Frames: ', n_frames)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        img_arr = frame[:, :, ::-1].transpose(2, 1, 0).astype(np.float32).copy()
        img_arr = torch.from_numpy(img_arr).to('cuda')
        img_arr = img_transform(img_arr).to('cpu').unsqueeze(0)

        video_list.append(img_arr)
    
    total_length = len(video_list)
    clamp_pos = random.randint(0, total_length - 1 - clip_length) if clip_length < total_length else 0
    video_list = video_list[clamp_pos: clamp_pos + clip_length]
    
    
    video_clip = torch.cat(video_list, axis=0)
    return video_clip

# transformation object
transform = Transform(IMG_SIZE, CROP_SIZE)


def read_todo(todo_path):
    """Reads TODO file to process videos
    """
    try:
        with open(todo_path) as f:
            d = [i.strip('\n') for i in f.readlines()]
        return d
    except:
        print ("File not found!")
        return []


def files_to_process(todo_path):
    processed_todo_path = todo_path.split(".txt")[0] + '_processed.txt'
    
    todo = read_todo(todo_path)
    proc = read_todo(processed_todo_path)
    
    d = list(set(todo) - set(proc))
    return d

def write_processed(todo_path, fname):
    """Writes processed video[s] here
    """
    processed_todo_path = todo_path.split(".txt")[0] + '_processed.txt'
    with open(processed_todo_path, 'a') as f:
        f.write(fname)
        f.write('\n')

def write_errors(todo_path, fname):
    processed_todo_path = todo_path.split(".txt")[0] + '_errors.txt'
    with open(processed_todo_path, 'a') as f:
        f.write(fname)
        f.write('\n')

def process_video(video_path):
    try:
        arr = extract_video_features(video_path, transform, CLIP_LENGTH)
        torch.save(arr, video_path.split('.mp4')[0] + '.pt')
        write_processed(todo_path, video_path)
    except Exception as err:
        print (err)
        write_errors(todo_path, video_path)
    
def process_videos(videos):
    for video_path in tqdm(videos):
        try:
            process_video(video_path)
        except Exception as err:
            print (err)
            write_errors(todo_path, video_path)
            
    
if __name__ == '__main__':
    args = sys.argv
    todo_path = args[1]
    videos = files_to_process(todo_path)
    
    print ("Files to process: ", len(videos))
    process_videos(videos)

    print ("Done.")
    

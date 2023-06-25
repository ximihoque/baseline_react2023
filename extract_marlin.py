import os
from marlin_pytorch import Marlin

from tqdm import tqdm
import torch
import os
import sys

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
        
def load_model(feature_type):
    model = Marlin.from_file(f"marlin_vit_{feature_type}_ytf", f"/home/surbhi/ximi/marlin_models/marlin_vit_{feature_type}_ytf.encoder.pt")
    return model

def process_video(video_path, model):
    try:
        arr = model.extract_video(video_path, crop_face=True)
        torch.save(arr, video_path.split('.mp4')[0] + '_marlin.pt')
        write_processed(todo_path, video_path)
    except Exception as err:
        print (err)
        write_errors(todo_path, video_path)
    
def process_videos(videos, model):
    for video_path in tqdm(videos):
        
            process_video(video_path, model)
        # except Exception as err:
        #     print (err)
        #     write_errors(todo_path, video_path)
def main(todo_path, model):
    videos = files_to_process(todo_path)
    print ("Files to process: ", len(videos))
    process_videos(videos, model)
    
if __name__ == '__main__':
    
    model = load_model('large')
    model = model.cuda()
    
    args = sys.argv
    todo_path = args[1]
    main(todo_path, model)
    
    



    

    print ("Done.")

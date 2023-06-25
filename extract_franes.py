from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import os
from tqdm import tqdm
import sys
from PIL import Image

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
def save_frame(arr, path):
    im = Image.fromarray(arr[:, :, ::-1])
    im.save(path)

def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
def read_frames(video_path):
    
    fvs = FileVideoStream(video_path).start()
    time.sleep(1)
    fps = FPS().start()

    cnt = 0
    while fvs.more():

        frame = fvs.read()
        yield frame

def video_to_frames(video_path):
    """
    Converts video to frames and saves them into respective directory
    """
    basename = os.path.basename(video_path).split('.')[0]
    dir_path = os.path.join(os.path.dirname(video_path), basename)

    # creating directories
    create_dirs(dir_path)
    
    frames = list(read_frames(video_path))[:-1]
    # writing to disk
    for idx, frame in enumerate(tqdm(frames)):
        path = os.path.join(dir_path, f'{idx}.png')
        save_frame(frame, path)

def process_videos(videos):
    for video_path in tqdm(videos):
        try:
            video_to_frames(video_path)
            write_processed(todo_path, video_path)
        except Exception as err:
            print (err)
            print (video_path)
            write_errors(todo_path, video_path)
if __name__ == '__main__':
    args = sys.argv
    todo_path = args[1]
    videos = files_to_process(todo_path)
    
    print ("Files to process: ", len(videos))
    process_videos(videos)

    print ("Done.")
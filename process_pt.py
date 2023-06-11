import torch
import numpy  as np
import glob
from tqdm import tqdm
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

if __name__ == '__main__':
    todo_path = sys.argv[1]
    todo = files_to_process(todo_path)
    print ("Files to process: ", len(todo))

    smaller_clips = []
    for f in tqdm(todo):
        arr = torch.load(f)
        if arr.shape[0] != 750:
            write_errors(todo_path, f)

        arr.to('cuda')
        torch.save(arr, f)
        write_processed(todo_path, f)


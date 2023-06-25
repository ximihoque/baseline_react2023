import torch
import numpy  as np
import glob
from tqdm import tqdm
import sys
from marlin_pytorch import Marlin
import torch
from torchvision import transforms
center = transforms.CenterCrop(224)

def load_model(feature_type):
    model = Marlin.from_file(f"marlin_vit_{feature_type}_ytf", f"/home/surbhi/ximi/marlin_models/marlin_vit_{feature_type}_ytf.encoder.pt")
    return model

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

model = load_model('large')
model = model.cuda()

def extract_marlin_features(tensor):
    tensor = center(tensor)
    arr = tensor.reshape(-1, 3, 224, 224)
    arr = arr.unfold(0, 16, 8)
    bs = 64
    features = []
    for i in range(0, arr.size(0), bs):
#         print (i)
        batch = arr[i: i+bs]
        with torch.no_grad():
            features.append(model.extract_features(batch.permute(0, 1, 4, 2, 3)).cpu())
        torch.cuda.empty_cache()
    return torch.cat(features).mean(0).cuda()

if __name__ == '__main__':
    todo_path = sys.argv[1]
    todo = files_to_process(todo_path)
    print ("Files to process: ", len(todo))

    smaller_clips = []
    for f in tqdm(todo):
        try:
            arr = torch.load(f).reshape(-1, 3, 256, 256)
            arr = extract_marlin_features(arr)

#             fname = f.split('.pt')[0] + '_marlin.pt'
#             print (fname)
            torch.save(arr, f)
            write_processed(todo_path, f)
        except Exception as err:
            print (f)
            print (err)
            write_errors(todo_path, f)


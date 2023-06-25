import os
from marlin_pytorch import Marlin

from tqdm import tqdm
import torch
import os
import sys
import glob
from PIL import Image
from torchvision import transforms

class Transform(object):
    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __call__(self, img):
        
        transform = transforms.Compose([
        
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
        
        ])
        img = transform(img)
        return img

transform = Transform(224)
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

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

def read_imgs(img_paths):
    clip = []
    for img in img_paths:
        im = pil_loader(img)
        clip.append(transform(im).unsqueeze(0))
    return torch.cat(clip).cuda()

def extract_marlin_features(dir_name):

    img_paths = glob.glob(os.path.join(dir_name, '*.png'))
    tensor = read_imgs(img_paths)
    
    arr = tensor.unfold(0, 16, 8)
    bs = 64
    features = []
    for i in range(0, arr.size(0), bs):
        batch = arr[i: i+bs]
        with torch.no_grad():
            features.append(model.extract_features(batch.permute(0, 1, 4, 2, 3)).cpu())
        torch.cuda.empty_cache()
    return torch.cat(features).mean(0).cuda()

def process_video(video_path, model):
    try:
        dir_name = video_path.split('.mp4')[0]
        arr = extract_marlin_features(dir_name)
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
if __name__ == '__main__':
    
    model = load_model('large')
    model = model.cuda()
    
    args = sys.argv
    todo_path = args[1]
    videos = files_to_process(todo_path)
    
    print ("Files to process: ", len(videos))
    process_videos(videos, model)

    print ("Done.")

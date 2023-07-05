import os
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
from model import TransformerVAEEmtMarlin
from utils import AverageMeter
from render import Render
from dataset_emt import get_dataloader
from model.losses import VAELoss, div_loss, ContrastiveLoss

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="../data", type=str, help="dataset path")
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--optimizer-eps', default=1e-8, type=float)
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('--seq-len', default=751, type=int, help="length of clip")
    parser.add_argument('--max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--use-hubert', default=False, type=bool, help="Use HuBERT model")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--render', action='store_true', help='w/ or w/o render')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--div-p', default=10, type=float, help="hyperparameter for div-loss")
    parser.add_argument('--use-video',  default=False, action='store_true', help='w/ or w/o video modality')

    args = parser.parse_args()
    return args


# Validation
def val(args, model, val_loader, render):
    epoch = 0
    # model.eval()
    model.reset_window_size(8)
    for batch_idx, (speaker_video, speaker_video_clip_orig, _, speaker_emotion, _, _, _, listener_emotion, listener_3dmm, listener_references) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            speaker_emotion,  listener_emotion, listener_3dmm = \
                speaker_emotion.cuda(), listener_emotion.cuda(), listener_3dmm.cuda()

            if args.use_video:
                speaker_video = speaker_video.cuda()
                
            else:
                speaker_video = None
        with torch.no_grad():
            prediction = model(speaker_emotion=speaker_emotion, speaker_video=speaker_video, is_train=False)
            listener_3dmm_out, listener_emotion_out, distribution = prediction

            val_path = os.path.join(args.outdir, 'results_videos', 'val')
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            B = speaker_emotion.shape[0]
            if (batch_idx % 50) == 0:
                for bs in range(B):
                    render.rendering(val_path, "e{}_b{}_ind{}".format(str(epoch + 1), str(batch_idx + 1), str(bs + 1)),
                            listener_3dmm_out[bs], speaker_video_clip_orig[bs].cuda(), listener_references[bs].cuda())

    model.reset_window_size(args.window_size)


def main(args):
    start_epoch = 0

    # val dataloader
    if args.render:
        val_loader = get_dataloader(args, "data/render.csv", load_audio=False, load_video_s=args.use_video, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True, load_ref=True, load_video_orig=True, use_raw_audio=args.use_hubert, mode='val')
    
    model = TransformerVAEEmtMarlin(img_size = args.img_size, audio_dim = args.audio_dim,  output_3dmm_dim = args._3dmm_dim, output_emotion_dim = args.emotion_dim, feature_dim = args.feature_dim, 
    seq_len = args.seq_len, max_seq_len=args.max_seq_len, online = args.online, window_size = args.window_size, use_hubert=args.use_hubert, device = args.device)
  

    checkpoint_path = args.resume
    print("Resume from {}".format(checkpoint_path))
    checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        print ("Using GPU")
        model = model.cuda()
        render = Render('cuda')
    else:
        render = Render()

        
    val(args, model, val_loader, render)
    # print("Epoch:  {}   val_loss: {:.5f}   val_rec_loss: {:.5f}  val_kld_loss: {:.5f} ".format(1, val_loss, rec_loss, kld_loss))
    
# ---------------------------------------------------------------------------------

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    args = parse_arg()
    # os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)


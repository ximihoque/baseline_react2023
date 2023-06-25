# import dill as pickle
from dataset import ReactionDataset
from torch.utils.data import DataLoader
from model import TransformerVAE
import pdb
from tqdm import tqdm
# from model import HuBERTEncoder
import torch


print ('num threads: ', torch.get_num_threads())
torch.set_num_threads(1)
audio_flag = True
model = TransformerVAE(
    img_size=256,    
    use_hubert=audio_flag,
audio_dim=128, # becomes irrelevant in case for hubert
max_seq_len=751, 
seq_len=750)
model = model.cuda()
dataset = ReactionDataset('../data', 'data/sample_udiva.csv', clip_length=750, use_raw_audio=audio_flag, load_3dmm_l=True)
shuffle = True 
# sample = dataset[0]
# hubert = HuBERTEncoder(1024)
# hubert = hubert.cuda()

train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=2)
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    checkpoints = torch.load('./results/marlin_hubert/cur_checkpoint.pth')
    state_dict = checkpoints['state_dict']
    model.load_state_dict(state_dict)
    # speaker_video_clip, speaker_audio_clip, _, _, _, _, listener_emotion, listener_3dmm, listener_references = dataset[0]
    # print (listener_3dmm.shape)

    # print ("enumeratin")
    for batch_idx, (speaker_video_clip, _, speaker_audio_clip, _, _, _, _, listener_emotion, listener_3dmm, listener_references) in enumerate(tqdm(train_loader)):

        print ('speaker video shape: ', speaker_video_clip.shape)
        print ('speaker audio shape: ', speaker_audio_clip.shape)
        print ('listener 3d mm shape: ', listener_3dmm.shape)
        print ("running inference...")
        # out = hubert(speaker_audio_clip.cuda())
        
    # # pdb.set_trace();
        listener_3dmm_out, listener_emotion_out, distribution = model(speaker_video_clip.cuda(), speaker_audio_clip.cuda())
        pdb.set_trace()
        # print ('output shape: ', listener_3dmm_out.shape)
        # print ('dist shape: ', len(distribution))
    print ("done.")
    # print ('Batch idx: ', batch_idx)
    # train_loader = get_dataloader(args, "data/train.csv", load_ref=False, load_video_l=False)
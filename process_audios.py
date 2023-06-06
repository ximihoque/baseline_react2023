import moviepy.editor
import glob
from tqdm import tqdm
import os
import soundfile as sf
import torchaudio
import torch
import numpy as np


def extract_audio_features(audio_path, fps, n_frames):
    # video_id = osp.basename(audio_path)[:-4]
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs
        # rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        # zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)
        # curr_feat = np.concatenate((curr_mfccs, rms, zcr))

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_fname(f, ext=True):
    if ext:
        return f.split('/')[-1]
    else:
        return f.split('/')[-1].split('.')[0] 
    
# train = glob.glob('../UDIVA/Videos/**/**/**/*.mp4')
# val = glob.glob('../NoXI/val/videos/**/**/*.wav')
train = glob.glob('../data/combined/Audios/UDIVA/**/**/**/*.wav')

def process_audios(audios):
    """Creates .wav for video files
    """
    for path in tqdm(audios):
#         video = moviepy.editor.VideoFileClip(path)
        dir_name = os.path.dirname(path)
            #.replace('/Videos/', '/Audios/')
#         create_dir(dir_name)
        features = extract_audio_features(path, 25, 750)
        audio_path = os.path.join(dir_name, 
                                  get_fname(path, False) + '_audio.npy')
        np.save(audio_path, features)
#         if not os.path.exists(audio_path):
#             video.audio.write_audiofile(audio_path)


if __name__ == '__main__':
    process_audios(train)
#     process_audios(val)

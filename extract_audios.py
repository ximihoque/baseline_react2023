import sys
from tqdm import tqdm
import moviepy.editor
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

def extract_audio(video_path):
    video = moviepy.editor.VideoFileClip(video_path)
    audio_path = video_path.replace('Videos', 'Audios').split('.mp4')[0] + '.wav'
    # video.audio.write_audiofile(audio_path) 
    print ('video_path: ', video_path)
    print ('audio_path: ', audio_path)


if __name__ == '__main__':
    args = sys.argv

    video_paths = read_todo(args[1])
    for path in tqdm(video_paths):
        extract_audio(path)

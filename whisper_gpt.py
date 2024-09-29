import os
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
from safetensors import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import torch


class Whisper:
    def __init__(self, path_to_whisper: str):
        self.processor = WhisperProcessor.from_pretrained(
            path_to_whisper,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            path_to_whisper,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.model = self.model.to(self.device)
    
    @staticmethod
    def get_audio(path: str, output_path: str) -> bool:
        try:
            clip = VideoFileClip(path)
            if clip is None:
                return False
            
            audio = clip.audio
            if audio is None:
                clip.close()
                return False
            
            audio.write_audiofile(output_path)
            audio.close()
            clip.close()
            return True
        except Exception:
            return False
    
    def __call__(self, path_to_videos: str, is_tqdm: bool = False) -> dict[str, str]:
        os.mkdir('audio')
        for filename in os.listdir(path_to_videos):
            load_path = os.path.join(path_to_videos, filename)
            filename, _ = os.path.splitext(filename)
            
            mp3_filename = filename + '.mp3'
            wav_filename = filename + '.wav'
            
            mp3_path = os.path.join('audio', mp3_filename)
            wav_path = os.path.join('audio', wav_filename)
            
            if self.get_audio(load_path, mp3_path):
                audio, _ = librosa.load(mp3_path, sr=16000)
                sf.write(wav_path, audio, 16000, 'PCM_24')
                os.remove(mp3_path)
        
        dataset = load_dataset('audiofolder', data_dir='audio')['train']
        
        answer = {}
        
        circle = range(len(dataset)) if not is_tqdm else tqdm(range(len(dataset)))
        for i in circle:
            array = dataset[i]['audio']['array']
            name = dataset[i]['audio']['path']
            k = array.shape[0] // 100000
            audio_arrays = np.array_split(array, k)
            input_features = self.processor(audio_arrays, sampling_rate=16000, return_tensors="pt").input_features
            predicted_ids = self.model.generate(input_features.to(self.device))
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            answer[name] = ' '.join(transcription)
        
        os.system('rm -r audio')
        
        return answer
    
    
def main():
    os.system('rm -r audio')
    whisper_path = '/models/whisper_small_folder'
    data_path = '/src/test/single_video'
    
    process_videos = Whisper(whisper_path)
    result = process_videos(data_path)
    print(result)


if __name__ == '__main__':
    main()
"""
data.py: data handling implementation
-------------------------------------


* Copyright: 2023 Tampere University
* Authors: Khoa Pham Dinh (khoa.phamdinh@tuni.fi), Uyen Phan (uyen.phan@tuni.fi)
* Date: 2023-11-10
* Version: 0.0.1

This is part of the audio_processing project

License
-------
Proprietary License

"""

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import soundfile as sf
import librosa
import audiolazy

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def normalize_audio(file_path, output_path):
    # Load the audio file
    print(file_path)
    audio, sample_rate = librosa.load(file_path)

    # Normalize the audio values to [-1, 1]
    max_val = np.max(np.abs(audio))
    normalized_audio = audio / max_val

    # Stretch the audio to exactly 5 seconds
    # Using librosa.effects.time_stretch
    if len(normalized_audio) > sample_rate * 5:
        # Normalize the audio time length to 5 seconds
        normalized_audio = librosa.effects.time_stretch(y = normalized_audio,
                                                        rate = librosa.get_duration(y=normalized_audio, sr=sample_rate)/5)
    elif len(normalized_audio) < sample_rate * 5:
        # Stretch the audio to 5 seconds
        normalized_audio = librosa.effects.time_stretch(y = normalized_audio,
                                                        rate = 5/librosa.get_duration(y=normalized_audio, sr=sample_rate))

    print(len(normalized_audio))
    print(sample_rate)
    # Save the normalized audio
    sf.write(output_path, normalized_audio, sample_rate, format='wav')

    return normalized_audio

def normalized_audios(audio_dir):
    # Normalize the audios in the folder "audio_dir"
    # and save the normalized audios to the folder "normalized_audio_dir"
    normalized_audio_dir = os.path.join(DATA_DIR, 'normalized_' + audio_dir.split('/')[-1])
    if not os.path.exists(normalized_audio_dir):
        os.makedirs(normalized_audio_dir)

    for file in os.listdir(audio_dir):
        print(file)
        if not file.endswith('.wav'):
            continue

        file_path = os.path.join(audio_dir, file)
        output_path = os.path.join(normalized_audio_dir, file)
        normalize_audio(file_path, output_path)


if  __name__ == '__main__':

    # Genearte the normalized audios
    normalized_audios(os.path.join(DATA_DIR, 'test_tram'))
    normalized_audios(os.path.join(DATA_DIR, 'test_car'))
    normalized_audios(os.path.join(DATA_DIR, 'train_tram'))
    normalized_audios(os.path.join(DATA_DIR, 'train_car'))
    normalized_audios(os.path.join(DATA_DIR, 'valid_tram'))
    normalized_audios(os.path.join(DATA_DIR, 'valid_car'))

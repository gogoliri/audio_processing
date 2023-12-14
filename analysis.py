"""
analysis.py: Analysis
---------------------


* Copyright: 2023 Tampere University
* Authors: Khoa Pham Dinh (khoa.phamdinh@tuni.fi), Uyen Phan (uyen.phan@tuni.fi)
* Date: 2023-12-01
* Version: 0.0.1

This file calculates:
    energy,
    RMS,
    zero-crossing rate,
    log-spectrograms,
    logmel-spectrograms,
    CQT spectrograms,
    MFCCs.

License
-------
Proprietary License

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import soundfile as sf
import librosa
import audiolazy
import os

def signal_energy(y):
    """
    Energy
    """
    return np.sum(np.power(y, 2))


def signal_rms(y):
    """
    RMS
    """
    return np.sqrt(np.mean(np.power(y, 2)))


def signal_zero_crossing_rate(y):
    """
    Zero-crossing rate
    """
    return np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))


def log_spectrogram(y,
                    sr,
                    n_fft=1024,
                    hop_length=512,
                    win_length=1024,
                    window='hann',
                    is_plot=False,):
    """
    Log-spectrogram
    """
    D = librosa.stft(y,
                     n_fft=n_fft,
                     hop_length=hop_length,
                     win_length=win_length,
                     window=window,
                     center=True)

    log_S = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    if is_plot:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(log_S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log frequency power spectrogram')
        plt.show()

    return log_S


def logmel_spectrogram(y, sr, n_fft=1024, hop_length=512, win_length=1024, window='hann', n_mels=128, is_plot=False,):
    """
    Logmel-spectrogram
    """
    D = librosa.stft(y,
                     n_fft=n_fft,
                     hop_length=hop_length,
                     win_length=win_length,
                     window=window,
                     center=True)

    logmel_S = librosa.power_to_db(librosa.feature.melspectrogram(S=np.abs(D)**2, sr=sr, n_mels=n_mels))

    if is_plot:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(logmel_S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log mel-frequency power spectrogram')
        plt.show()

    return logmel_S


def cqt_spectrogram(y,
                    sr,
                    hop_length=512,
                    win_length=1024,
                    window='hann',
                    is_plot=False,):
    """
    CQT-spectrogram
    """
    C = librosa.cqt(y,
                    sr=sr,
                    hop_length=hop_length,
                    window=window,
                    )

    cqt_S = librosa.amplitude_to_db(np.abs(C), ref=np.max)

    if is_plot:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(cqt_S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q transform power spectrogram')
        plt.show()

    return cqt_S


def mfcc(y,
         sr,
         hop_length=512,
         win_length=1024,
         n_mfcc=13,
         n_fft=1024,
         window='hann',
         is_plot=False,):
    """
    MFCC
    """
    mfcc = librosa.feature.mfcc(y=y,
                                sr=sr,
                                hop_length=hop_length,
                                win_length=win_length,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                window=window,
                                center=True,)

    if is_plot:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis='time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('MFCC coefficients')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency cepstral coefficients')
        plt.show()

    return mfcc


def feature_extraction(audio_file,
                       is_energy=True,
                       is_rms=True,
                       is_zcr=True,
                       is_log_spectrogram=True,
                       is_logmel_spectrogram=True,
                       is_cqt_spectrogram=True,
                       is_mfcc=True,
                       is_plot=True,
                       n_fft=1024,
                       win_length=1024,
                       hop_length=512,
                       window='hann',
                       ):
    "Given audio file, extract features"
    # Read audio file
    y, sr = librosa.load(audio_file)

    if is_plot:
        # Plot signal
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr=sr, axis='time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title('Signal')
        plt.show()
    # Extract features
    features = {}
    if is_energy:
        energy = signal_energy(y)
        features['energy'] = energy
    else:
        features['energy'] = None

    if is_rms:
        rms = signal_rms(y)
        features['rms'] = rms
    else:
        features['rms'] = None

    if is_zcr:
        zcr = signal_zero_crossing_rate(y)
        features['zcr'] = zcr
    else:
        features['zcr'] = None

    if is_log_spectrogram:
        log_S = log_spectrogram(y, sr, is_plot=is_plot)
        features['log_spectrogram'] = log_S
    else:
        features['log_spectrogram'] = None

    if is_logmel_spectrogram:
        logmel_S = logmel_spectrogram(y, sr, is_plot=is_plot)
        features['logmel_spectrogram'] = logmel_S
    else:
        features['logmel_spectrogram'] = None

    if is_cqt_spectrogram:
        cqt_S = cqt_spectrogram(y, sr, is_plot=is_plot)
        features['cqt_spectrogram'] = cqt_S
    else:
        features['cqt_spectrogram'] = None

    if is_mfcc:
        MFCC = mfcc(y, sr, is_plot=is_plot)
        features['mfcc'] = MFCC
    else:
        features['mfcc'] = None

    return features

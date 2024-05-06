import sys
import numpy as np
from os import listdir
from pathlib import Path
from scipy.io import wavfile
import librosa
from sklearn.preprocessing import LabelBinarizer
from src import pre_processing

def get_data(dataset_path, FULL_DATA=False, feature='mfcc'):
    path_music = str(dataset_path) + "/genres_original/"
    genres = listdir(path_music)

    # Extract features
    X, fs = __read_audio_data_3sec(path_music)
    #X = __mellog_spectra (X, fs)

    # Labels
    y = []
    i = 0
    for genere in genres:
        n = len(listdir(path_music+genere))
        y += [genere]*n
    y = __encode_labels(y)

    
    return X, y

def __read_audio_data(path_music):
    # Sampling frequency & length of the file
    fs = librosa.load('Data/genres_original/blues/blues.00000.wav')[1]
    sz = 661500 #len(wavfile.read('Data/genres_original/blues/blues.00000.wav')[1])

    genres = listdir(path_music)
    
    # Pre-allocate the memory
    data = np.zeros((1000,sz))

    # Read files 
    sys.stdout.write("> Leggo i file audio...\n")
    cont = 0
    for i in range(len(genres)):
        sys.stdout.write("--> Leggo "+genres[i]+"...\n")
        genre_path = (path_music+genres[i])
        sounds = listdir(genre_path)
        for y in range(len(sounds)):
            sample = librosa.load(genre_path+'/'+sounds[y])[0]
            if(len(sample) >= sz):
                data[cont,:] = sample[:sz]
            else:
                data[cont,:] = np.hstack([sample, np.zeros(sz-len(sample))])
            #data[cont,:] = librosa.load(genre_path+'/'+sounds[y])[0][:sz]
            cont = cont + 1

    return data, fs

def __read_audio_data_3sec(path_music):
    fs = librosa.load('Data/genres_original/blues/blues.00000.wav')[1]
    sz = 661500 #len(wavfile.read('Data/genres_original/blues/blues.00000.wav')[1])
    genres = listdir(path_music)
    data = np.zeros((10000,66150))

    cont = 0
    for i in range(len(genres)):
        genre_path = (path_music+genres[i])
        sounds = listdir(genre_path)
        for y in range(len(sounds)):
            sample = librosa.load(genre_path+'/'+sounds[y])[0]
            if(len(sample) >= sz):
                x = sample[:sz]
            else:
                x = np.hstack([sample, np.zeros(sz-len(sample))])
            x_split = np.array_split(x,10)
            for j in x_split:
                data[cont,:] = j
                cont = cont + 1

    return data, fs


def __mellog_spectra(data, fs):
    """Calculate mel-log spectra from the given audio data"""
    
    n_samples = data.shape[0]
    #spectra = np.zeros((n_samples, 128, 1292))
    spectra = np.zeros((n_samples, 128, 130))
    
    sys.stdout.write("\nCalculating spectra...\n")
    for i in range(n_samples):
        s = librosa.feature.melspectrogram(data[i,:], sr=fs)  # mel-spectrogram
        s[s!=0] = np.log(s[s!=0])  # logarithm
        spectra[i,:,:] = s  # add to data
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    return spectra


def __encode_labels(y):
    """Encode targets into onehot-form. 
    Also save the string-type labels for future decoding"""
    lb = LabelBinarizer()
    lb.fit(y)
    np.save(Path('tmp') / 'classes.npy', lb.classes_)
    return lb.transform(y)    

if __name__ == '__main__':
    DATASET_path = Path('Data')
    X, y = get_data(DATASET_path, FULL_DATA=True, feature='mel')
    X = __mellog_spectra(X,22050)
   

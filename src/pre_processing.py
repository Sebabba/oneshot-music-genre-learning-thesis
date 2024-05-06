import sys
import numpy as np
from os import listdir
from pathlib import Path
from scipy.io import wavfile
import librosa
from sklearn.preprocessing import LabelBinarizer


def get_data(dataset_path, FULL_DATA=False, feature='mel', smp_lenght=3):
    path_music = str(dataset_path) + "/genres_original/"
    genres = listdir(path_music)

    # Extract features
    if Path(feature + '_log_spectra.npy').is_file():
        X = np.load(feature + '_log_spectra.npy')
    else:
        if smp_lenght==30:
            # 30s samples
            X, fs = __read_audio_data(path_music)
        else:
            # 3s samples
            X, fs = __read_audio_data_3sec(path_music)

        if feature == 'mel':
            X = __mellog_spectra (X, fs, smp_lenght)
        elif feature == 'mfcc':
            X = __mfcc_spectra(X, fs)
        else:
            X = __constantQ_spectra(X, fs)

        np.save(feature + '_log_spectra', X)

    # Labels
    y = []
    i = 0
    for genere in genres:
        n = len(listdir(path_music+genere))
        if smp_lenght==3:
            n = n*10

        y += [genere]*n
    #y = __encode_labels(y)


    return X, y


def class_split(X, y, split_size=0.5, seed=None):
    """Split data based on classes
        Inputs:
            X: Spectra for splitting
            y: Corresponding labels
            split_size: The size of the first splitted batch
            seed: Use a known seed for the rng
        Outputs:
            X_1: Splitted spectra with split_size from the original
            X_2: Splitted spectra with 1-split_size from the original
            y_1: Corresponding labels
            y_2: Corresponding labels
            """

    # Set seed if specified
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Split classes
    class_ids = np.nonzero(np.sum(y, axis=0))[0]
    n_classes = len(class_ids)
    rng.shuffle(class_ids)
    n_1 = int(split_size*n_classes)

    classes_1 = class_ids[:n_1]
    classes_2 = class_ids[n_1:]

    classes_1_ids = np.where(y[:,classes_1])[0]
    classes_2_ids = np.where(y[:,classes_2])[0]

    X_1 = X[classes_1_ids, :, :]
    X_2 = X[classes_2_ids, :, :]

    y_1 = y[classes_1_ids, :]
    y_2 = y[classes_2_ids, :]

    return X_1, X_2, y_1, y_2


# The split function is applied in every class
def class_split2(X, y, split_size=0.5, seed=None):
    """Split based on samples, inside the classes"""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    class_ids = np.nonzero(np.sum(y, axis=0))[0]
    n_classes = len(class_ids)

    class_1_ids = []
    class_2_ids = []

    for i in class_ids:
        cl = np.where(y[:,i])[0]
        rng.shuffle(cl)
        n_1 = int(split_size*len(cl))
        class_1_ids.extend(cl[:n_1])
        class_2_ids.extend(cl[n_1:])

    X_1 = X[class_1_ids, :, :]
    X_2 = X[class_2_ids, :, :]

    y_1 = y[class_1_ids, :]
    y_2 = y[class_2_ids, :]

    return X_1, X_2, y_1, y_2


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

# Mel-scaled spectrogram
def __mellog_spectra(data, fs, smp_lenght=30):
    """Calculate mel-log spectra from the given audio data"""

    n_samples = data.shape[0]
    if smp_lenght==30:
        # 30s samples
        spectra = np.zeros((n_samples, 128, 1292))
    else:
        # 3s samples
        spectra = np.zeros((n_samples, 128, 130))

    sys.stdout.write("\nCalculating spectra...\n")
    for i in range(n_samples):
        s = librosa.feature.melspectrogram(data[i,:], sr=fs)  # mel-spectrogram
        s[s!=0] = np.log(s[s!=0])  # logarithm
        spectra[i,:,:] = s  # add to data
        __update_progressbar(i/(n_samples-1))  # update progress
    sys.stdout.write("\n")
    sys.stdout.flush()

    return spectra

# MFCC
def __mfcc_spectra(data, fs):
    n_samples = data.shape[0]
    spectra = np.zeros((n_samples, 40, 1292))

    sys.stdout.write("\n>Calcolo gli MFCC dei vari audio...\n")
    for i in range(n_samples):
        s = librosa.feature.mfcc(data[i,:], fs, S=None, n_mfcc=40)
        spectra[i,:,:] = s
        __update_progressbar(i/(n_samples-1))

    sys.stdout.write("\n")
    sys.stdout.flush()
    return spectra

# Constant-Q chromagram
def __constantQ_spectra(data, fs):
    n_samples = data.shape[0]
    spectra = np.zeros((n_samples, 84, 1292))

    sys.stdout.write("\nCalcolo i Constant-Q dei vari audio...\n")
    for i in range(n_samples):
        s = librosa.feature.chroma_cqt(y=data[i,:], sr=fs, n_chroma=40, bins_per_octave=80)
        spectra[i,22:62,:] = s
        __update_progressbar(i/(n_samples-1))

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


def __update_progressbar(progress):
    """ Display a progress bar with the given amount of progress"""

    # The length of the progress bar
    bar_length = 40

    # Make sure progress is float and not over 1
    progress = float(min(progress, 1))

    # Number of blocks currently shown
    block = int(round(bar_length*progress))

    # Make a progress bar
    text = "\rProgress: [{}] {:4.1f}%".format( "#"*block + "-"*(bar_length-block), progress*100)

    # Display
    sys.stdout.write(text)
    sys.stdout.flush()

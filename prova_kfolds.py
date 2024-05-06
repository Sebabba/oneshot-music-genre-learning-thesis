import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICE"]="1";

import sys
from pathlib import Path

import numpy as np
############ Keras/Tensorflow randomness ###################
from numpy.random import seed
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from src import model, pre_processing
from src.PairGenerator import PairGenerator

from sklearn.model_selection import StratifiedKFold

seed(123)
#from tensorflow import set_random_seed
import tensorflow

tensorflow.random.set_seed(234)

DATASET_path = Path('Data')


if __name__ == '__main__':
    X, y = pre_processing.get_data(DATASET_path, FULL_DATA=True, feature='mel', smp_lenght=30)
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1336)
    # bisogna prima effettuare lo split e poi fare l'__encode delle label
    for xtr,xte in folds.split(X,y):
        print("--------------------------------------------------")
        print("xte, indici")
        rng = np.random.RandomState(1)
        rng.shuffle(xte)
        print(xte)
        h = [y[i] for i in xte]
        

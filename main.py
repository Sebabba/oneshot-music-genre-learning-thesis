import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICE"]="1";

from pathlib import Path
import numpy as np
from src import pre_processing, model
from src.PairGenerator import PairGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import sys


############ Keras/Tensorflow randomness ###################
from numpy.random import seed
seed(123)
#from tensorflow import set_random_seed
import tensorflow
tensorflow.random.set_seed(234)
############################################################


# Path to ESC-50 master directory (change if needed)
DATASET_path = Path('Data')


if __name__ == '__main__':
    # Load inputs, targets and metadata
    X, y = pre_processing.get_data(DATASET_path, FULL_DATA=True, feature='mel', smp_lenght=30)    
    sys.exit()
    n_test = 1
    ## 1 = 60%, 20%, 20%
    ## 2 = 40%, 20%, 20%
    ## 3 = 20%, 20%, 20%
    ## 4 = 10%, 10%, 20%

    X_train, X, y_train, y = pre_processing.class_split2(X, y, split_size=0.6, seed=1)
    X_oneshot_val, X_oneshot_test, y_oneshot_val, y_oneshot_test = pre_processing.class_split2(X, y, split_size=0.5, seed=3)
    
    if n_test == 2:
        X_train, X, y_train, y = pre_processing.class_split2(X_train, y_train, split_size=0.66667, seed=1)
    elif n_test == 3:
        X_train, X, y_train, y = pre_processing.class_split2(X_train, y_train, split_size=0.33334, seed=1)
    elif n_test == 4:
        X_train, X, y_train, y = pre_processing.class_split2(X_train, y_train, split_size=0.16667, seed=1)
        X,y = None, None  # release memory
        X_oneshot_val, X, y_oneshot_val, y = pre_processing.class_split2(X_oneshot_val, y_oneshot_val, split_size=0.5, seed=3)
        
    X,y = None, None  # release memory

    # Additional axis for 2D Conv
    X_train = X_train[:,:,:,np.newaxis]
    X_oneshot_test = X_oneshot_test[:,:,:,np.newaxis]
    X_oneshot_val = X_oneshot_val[:,:,:,np.newaxis]

    # Make sure the session is clear
    K.clear_session()
    
    # Create a data generator
    batch_size = 8  # change according to your GPU limitations std=20
    pairs = 1.0 # amount of pairs from the maximum
    neg_pair_ratio = 2.0  # negative-to-positive-pair ratio
    train_generator = PairGenerator(X_train, y_train, seed=1, 
                              amount_of_pairs=pairs, neg_pair_ratio=neg_pair_ratio,
                              batch_size=batch_size)    
    
    checkpoint_path = Path('checkpoint')
    
    try:
        mdl = load_model(str(checkpoint_path / 'model.h5'))
        print("Continuing training")
    except:
        print("Starting training from scratch")
        #Create a siamese network model and fit
		# Default parameters are chosen based on the hyperparameter 
		# optimization process
        mdl = model.create_siamese_model(input_shape=X_oneshot_val.shape[1:]) #[1:]
									   
    # Setup callbacks    
    patience = 0
    patience_thresh = 25
    best_so_far = 0.0
    
    for i in range(100): #std=20
        print("Iteration: {}".format(i))        
        history = mdl.fit(x=train_generator,
                          epochs=1, verbose=1)  
        
        score = model.test_oneshot(mdl, X_oneshot_val, y_oneshot_val,
                          visualize=False, save_results=False, seed=1, n_samples_per_classes=20) # if n_test == 4, n_samples_per_classes must be = 10

        # Save best score
        if score > best_so_far:
            mdl.save_weights(checkpoint_path / 'best_weights.h5')
            best_so_far = score
            patience = 0
        else:
            patience += 1
        
        if patience > patience_thresh:
            break
        
    # Load best weights
    mdl.load_weights(checkpoint_path / 'best_weights.h5')
    
    # Perform oneshot test
    print("One-shot test")
    model.test_oneshot(mdl, X_oneshot_test, y_oneshot_test,
                          visualize=False, save_results=True, seed=1)
    mdl.save(str(checkpoint_path / 'model.h5'))  # save for future training

    # Clear tmp folder
    for file in os.listdir(Path('tmp')):
        os.remove(Path('tmp') / file)
    del mdl
    K.clear_session()
    
    
    

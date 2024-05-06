import tensorflow as tf
import tensorflow.keras as keras

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from src import pre_processing, model

from tensorflow.keras.models import load_model

model_path = 'checkpoint/model.h5'
model = load_model(model_path)

# samples from X_oneshot_test set
smp1 = 84
smp2 = 81

DATASET_path = Path('Data')
X, y = pre_processing.get_data(DATASET_path, FULL_DATA=True, feature='mel')

X_train, X, y_train, y = pre_processing.class_split2(X, y, split_size=0.6, seed=1)
X_oneshot_val, X_oneshot_test, y_oneshot_val, y_oneshot_test = pre_processing.class_split2(X, y, split_size=0.5, seed=3)

X,y = None, None  # release memory

# Additional axis for 2D Conv
X_train = X_train[:,:,:,np.newaxis]
X_oneshot_test = X_oneshot_test[:,:,:,np.newaxis]
X_oneshot_val = X_oneshot_val[:,:,:,np.newaxis]

image1 = X_oneshot_test[smp1]
image1 = np.expand_dims(image1, axis=0)

image2 = X_oneshot_test[smp2]
image2 = np.expand_dims(image2, axis=0)

y_pred = model.predict([image1, image2])

images = [tf.Variable(image1, dtype=float), tf.Variable(image2, dtype=float)]

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]

grads = tape.gradient(loss, images)

dgrad_abs = tf.math.abs(grads)

dgrad_max_ = np.max(dgrad_abs, axis=4)[0][0]

dgrad_max_1 = np.max(dgrad_abs, axis=4)[1][0] #

## normalize to range between 0 and 1
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

grad_eval2 = (dgrad_max_1 - arr_min) / (arr_max - arr_min + 1e-18) #

fig, axes = plt.subplots(2,1,figsize=(15,5))

# smp1, first input
axes[0].imshow(image1[0,:,:,:])
axes[0].invert_yaxis()

# smp2, second input
##axes[2].imshow(image2[0,:,:,:])
##axes[2].invert_yaxis()

# saliency map of smp1
i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
axes[1].invert_yaxis()

# saliency map of smp2
##i = axes[3].imshow(grad_eval2,cmap="jet",alpha=0.8)
##axes[3].invert_yaxis()

fig.suptitle("Hiphop", fontsize=16)
plt.show()

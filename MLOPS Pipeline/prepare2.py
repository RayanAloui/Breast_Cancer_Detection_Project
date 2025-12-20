import tensorflow as tf
import pandas as pd

def prepare2(data_filename):
    #Images Datasets
    IMG_SIZE = 50 # Image resolution (small to reduce computation)

    #maps filename to actual image data
    def load_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_png(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1]
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((df_train_images["filename"], df_train_images["label"]))
    train_ds = train_ds.map(load_preprocess).shuffle(5000).batch(64).prefetch(1) #shuffle pour rendre les images aleatoires chaque epoch

    val_ds = tf.data.Dataset.from_tensor_slices((df_val_images["filename"], df_val_images["label"]))
    val_ds = val_ds.map(load_preprocess).batch(64).prefetch(1)

    test_ds = tf.data.Dataset.from_tensor_slices((df_test_images["filename"], df_test_images["label"]))
    test_ds = test_ds.map(load_preprocess).batch(64).prefetch(1)

    return train_ds, val_ds,test_ds

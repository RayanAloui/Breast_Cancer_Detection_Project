import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

IMG_SIZE = 224   # à adapter selon ton dataset
EPOCHS = 10

def train_model2(train_ds, val_ds):
    cnn_model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(
        optimizer=optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    return cnn_model, history

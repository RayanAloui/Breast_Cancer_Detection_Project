import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # À mettre en tout premier

import time
import numpy as np
import tensorflow as tf         # Importer après avoir réglé TF_CPP_MIN_LOG_LEVEL
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier



def train_model1(X_train, y_train, X_test, y_test):

    print("=== SOFTMAX REGRESSION - MODELING PHASE ===")
    print("=" * 65)

    # Data
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_train = y_train.values.astype(np.int32)
    y_test = y_test.values.astype(np.int32)

    # Convert labels to one-hot encoding for softmax
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=2)

    # Hyperparameters EXACTLY from Table 1 of the article
    hyperparams = {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'epochs': 3000
    }

    print(f"\n Hyperparameters (Table 1 - Exact from article):")
    for key, value in hyperparams.items():
        print(f"   {key}: {value}")

    # Model parameters
    input_dim = X_train.shape[1]
    output_dim = 2

    print(f"\n Building Softmax Regression model...")

    # Initialize weights and biases
    W = tf.Variable(
        tf.random.normal([input_dim, output_dim], stddev=0.01),
        name='weights'
    )
    b = tf.Variable(tf.zeros([output_dim]), name='bias')

    # Softmax function
    def softmax_model(X):
        logits = tf.matmul(X, W) + b
        return tf.nn.softmax(logits)

    # Cross-entropy loss (Equation 15)
    def cross_entropy_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
        return -tf.reduce_mean(
            tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        )

    # Predict class
    def predict_class(y_pred_proba):
        return tf.argmax(y_pred_proba, axis=1)

    # Optimizer (SGD)
    optimizer = tf.optimizers.SGD(
        learning_rate=hyperparams['learning_rate']
    )

    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / hyperparams['batch_size']))

    train_loss_history = []
    train_acc_history = []
    test_acc_history = []

    start_time = time.time()

    # Training loop
    for epoch in range(hyperparams['epochs']):
        epoch_loss = 0.0

        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]

        for batch in range(n_batches):
            start_idx = batch * hyperparams['batch_size']
            end_idx = min(
                (batch + 1) * hyperparams['batch_size'],
                n_samples
            )

            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            with tf.GradientTape() as tape:
                y_pred_proba = softmax_model(batch_X)
                loss = cross_entropy_loss(batch_y, y_pred_proba)

            gradients = tape.gradient(loss, [W, b])
            optimizer.apply_gradients(zip(gradients, [W, b]))

            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / n_batches
        train_loss_history.append(avg_loss)

        if (epoch + 1) % 500 == 0 or epoch == 0:
            train_pred = predict_class(
                softmax_model(X_train)
            ).numpy()
            train_acc = np.mean(train_pred == y_train)
            train_acc_history.append(train_acc)

            test_pred = predict_class(
                softmax_model(X_test)
            ).numpy()
            test_acc = np.mean(test_pred == y_test)
            test_acc_history.append(test_acc)

    training_time = time.time() - start_time

    total_data_points = (
        hyperparams['epochs']
        * hyperparams['batch_size']
        * n_batches
    )

    final_pred_proba = softmax_model(X_test)
    final_pred = predict_class(final_pred_proba).numpy()
    final_accuracy = np.mean(final_pred == y_test)

    return final_pred_proba, final_pred, final_accuracy, total_data_points, hyperparams['epochs']

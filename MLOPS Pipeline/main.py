import os
import tensorflow as tf

# Limiter les messages TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR

# Optionnel : désactiver les logs d’avertissement Python
import warnings
warnings.filterwarnings("ignore")
import time
import joblib
import argparse
import psutil  # Pour le monitoring système
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from prepare1 import prepare1
from prepare2 import prepare2
from prepare3 import prepare3
from modeling1 import train_model1
from modeling2 import train_model2
from modeling3 import train_model3
from evaluate1 import evaluate_model1
from evaluate2 import evaluate_model2
from evaluate3 import evaluate_model3
#from save_model import save_model
#from charger_model import charger_model

import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch, exceptions as es_exceptions

def main():
    parser = argparse.ArgumentParser(description="Pipeline Churn Modelling")
    parser.add_argument("--runDSO1", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--runDSO2", action="store_true", help="Tester le modèle")
    parser.add_argument("--runDSO3", action="store_true", help="Tester le modèle")
    parser.add_argument("--jareb", action="store_true", help="Tester le modèle")
    args = parser.parse_args()

    if args.runDSO1:
        X_train, X_test, y_train, y_test = prepare1("data.csv")
        y_proba, y_pred, acc, total_points, epochs = train_model1(X_train, y_train, X_test, y_test)
        evaluate_model1(y_test, y_proba, y_pred, acc, total_points, epochs)
    elif args.runDSO2:
        train_ds, val_ds,test_ds = prepare2("data_file")
        cnn_model, history = train_model2(train_ds, val_ds)
        evaluate_model2(cnn_model, test_ds, model_name="CNN")
    elif args.runDSO3:
        X_train3, X_test3, y_train3, y_test3 = prepare3("data2.csv")
        rf = train_model3(X_train3, y_train3, X_test3)
        evaluate_model3(rf, X_test3, y_test3)

# ==========================================================
# === Lancement ===
# ==========================================================

if __name__ == "__main__":
    main()


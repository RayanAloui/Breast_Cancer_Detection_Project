# =========================
# Imports nécessaires
# =========================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc
)

# =========================
# Fonction d'évaluation
# =========================
def evaluate_model2(model, test_ds, model_name="Model"):
    # Évaluation globale
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"{model_name} - Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    # Prédictions
    y_true = []
    y_pred = []
    y_scores = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).ravel()
        y_scores.extend(preds)
        y_pred.extend((preds > 0.5).astype(int))
        y_true.extend(labels.numpy())

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-IDC', 'IDC'],
        yticklabels=['Non-IDC', 'IDC']
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # =========================
    # Classification Report
    # =========================
    print(f"{model_name} - Classification Report:")
    print(classification_report(y_true, y_pred))

    # =========================
    # ROC Curve & AUC
    # =========================
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.show()

    #return {
    #    "accuracy": acc,
    #    "auc": roc_auc,
    #    "confusion_matrix": cm
    #}

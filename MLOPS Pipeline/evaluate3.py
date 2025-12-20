import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, roc_curve
)

def evaluate_model3(model,X_test, y_test, model_name="Model"):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    # 2. Metrics
    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"\n--- {model_name} RESULTS ---")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"AUC      : {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    # 3. Classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        pred,
        target_names=['Healthy (0)', 'Patient (1)']
    ))

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, pred)
    print("\nCONFUSION MATRIX:")
    print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

    # 5. Clinical metrics
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    print(f"\nCLINICAL METRICS:")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

    # 6. ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        label=f'{model_name} (AUC = {auc:.3f})',
        color='darkorange',
        linewidth=2
    )
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

#    return {
#       "pred": pred,
#        "proba": proba,
#        "accuracy": accuracy,
#        "auc": auc,
#        "precision": precision,
#        "recall": recall,
#        "f1": f1,
#        "sensitivity": sensitivity,
#        "specificity": specificity,
#        "confusion_matrix": cm
#    }

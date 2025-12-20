import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report,
    roc_curve
)

def evaluate_model1(
    y_test,
    final_pred_proba,
    final_pred,
    final_accuracy,
    total_data_points,
    n_epochs
):
    # Confusion matrix
    cm = confusion_matrix(y_test, final_pred)
    tn, fp, fn, tp = cm.ravel()

    # Clinical metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n Performance Metrics:")
    print(f"   Accuracy: {final_accuracy*100:.6f}% ({final_accuracy:.6f})")
    print(f"   FPR (False Positive Rate): {fpr*100:.6f}% ({fpr:.6f})")
    print(f"   FNR (False Negative Rate): {fnr*100:.6f}% ({fnr:.6f})")
    print(f"   TPR (True Positive Rate): {tpr*100:.6f}% ({tpr:.6f})")
    print(f"   TNR (True Negative Rate): {tnr*100:.6f}% ({tnr:.6f})")

    print(f"\n Confusion Matrix:")
    print(f"               Predicted 0   Predicted 1")
    print(f"   Actual 0       {tn:6d}         {fp:6d}")
    print(f"   Actual 1       {fn:6d}         {tp:6d}")

    # Comparison with article results (Table 2)
    print(f"\n" + "="*65)
    print("COMPARISON WITH ARTICLE RESULTS (Table 2)")
    print("="*65)

    article_results = {
        'Accuracy': 0.9765625,
        'FPR': 0.05769231,
        'FNR': 0.0,
        'TPR': 1.0,
        'TNR': 0.94230769,
        'Data Points': 384000,
        'Epochs': 3000
    }

    our_results = {
        'Accuracy': final_accuracy,
        'FPR': fpr,
        'FNR': fnr,
        'TPR': tpr,
        'TNR': tnr,
        'Data Points': total_data_points,
        'Epochs': n_epochs
    }

    print(f"\n   {'Metric':<20} {'Our Result':<15} {'Article':<15} {'Difference':<15}")
    print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*15}")

    for metric in ['Accuracy', 'FPR', 'FNR', 'TPR', 'TNR']:
        article_val = article_results[metric]
        our_val = our_results[metric]
        diff = our_val - article_val

        if metric == 'Accuracy':
            print(f"   {metric:<20} {our_val*100:>7.4f}%{'':<7} {article_val*100:>7.4f}%{'':<7} {diff*100:>+8.4f}%")
        else:
            print(f"   {metric:<20} {our_val:>8.6f}    {article_val:>8.6f}    {diff:>+8.6f}")

    print(f"\n   {'Data Points':<20} {total_data_points:>15} {article_results['Data Points']:>15} {total_data_points - article_results['Data Points']:>+15}")
    print(f"   {'Epochs':<20} {n_epochs:>15} {article_results['Epochs']:>15} {n_epochs - article_results['Epochs']:>+15}")

    # Performance summary
    print(f"\n" + "="*65)
    print("CONCLUSION")
    print("="*65)

    accuracy_diff = (final_accuracy - article_results['Accuracy']) * 100

    if accuracy_diff > 0:
        conclusion = f" BETTER than article by {accuracy_diff:+.4f}%"
    elif accuracy_diff < 0:
        conclusion = f" WORSE than article by {abs(accuracy_diff):.4f}%"
    else:
        conclusion = " EQUAL to article results"

    print(f"\n Softmax Regression Results:")
    print(f"   Our Accuracy:     {final_accuracy*100:.4f}%")
    print(f"   Article Accuracy: {article_results['Accuracy']*100:.4f}%")
    print(f"   {conclusion}")

    # Optionally, return metrics as a dictionary
    results_dict = {
        'accuracy': final_accuracy,
        'fpr': fpr,
        'fnr': fnr,
        'tpr': tpr,
        'tnr': tnr,
        'confusion_matrix': cm.tolist(),
        'data_points': total_data_points,
        'epochs': n_epochs,
        'accuracy_diff_percent': accuracy_diff,
        'better_than_article': final_accuracy > article_results['Accuracy']
    }

    return results_dict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_model3(X_train_bal, y_train_bal, X_test_bal):
    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Entraînement
    print("Random Forest Training...")
    rf.fit(X_train_bal, y_train_bal)

    return rf

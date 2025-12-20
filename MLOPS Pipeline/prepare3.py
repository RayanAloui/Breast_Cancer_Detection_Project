import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

def prepare3(data_filename):
    # Charger le dataset
    df = pd.read_csv(data_filename)
    
    # Copie pour traitement
    df_prep = df.copy()  # Remplacé clinical_df par df
    
    # Map target variable: 1=Healthy→0, 2=Patients→1
    df_prep['Classification'] = df_prep['Classification'].map({1: 0, 2: 1})
    
    # Normaliser les noms de colonnes
    df_prep.columns = df_prep.columns.str.strip()
    df_prep.columns = df_prep.columns.str.replace('.', '_', regex=False)
    
    # Renommer pour plus de clarté si nécessaire
    df_prep = df_prep.rename(columns={
        'MCP_1': 'MCP_1',  # Si besoin
        'HOMA': 'HOMA_IR'  # HOMA-IR est plus clair
    })
    
    # Séparer features et target
    X = df_prep.drop('Classification', axis=1)
    y = df_prep['Classification']
    
    # Création de nouvelles features
    X_enhanced = X.copy()
    
    # Ratios cliniquement pertinents
    X_enhanced['BMI_Glucose'] = X['BMI'] * X['Glucose'] / 100
    X_enhanced['HOMA_Leptin'] = X['HOMA_IR'] * X['Leptin'] / 100
    X_enhanced['Insulin_Resistin'] = X['Insulin'] * X['Resistin'] / 100
    X_enhanced['Adiponectin_Leptin_ratio'] = X['Adiponectin'] / (X['Leptin'] + 1e-10)
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enhanced)
    X_scaled = pd.DataFrame(X_scaled, columns=X_enhanced.columns, index=X_enhanced.index)
    
    # Option 1: SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
    
    # Option 2: Combinaison SMOTE + Undersampling
    smote_tomek = SMOTETomek(random_state=42)
    X_balanced, y_balanced = smote_tomek.fit_resample(X_scaled, y)
    
    # Split des données équilibrées
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
        X_balanced, y_balanced,
        test_size=0.25,
        random_state=42,
        stratify=y_balanced
    )
    
    return X_train_bal, X_test_bal, y_train_bal, y_test_bal

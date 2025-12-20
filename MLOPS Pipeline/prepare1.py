import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def prepare1(data_filename):
    df = pd.read_csv(data_filename)

    # Remove empty column if exists
    if df.columns[-1] == 'NaN' or pd.isna(df.columns[-1]):
        df = df.iloc[:, :-1]

    # Identify target column (assuming it's the second column as per WDBC standard)
    target_column = 'diagnosis' if 'diagnosis' in df.columns else df.columns[1]

    # Encode target variable: M=1, B=0
    df[target_column] = df[target_column].map({'M': 1, 'B': 0})

    # Separate features and target
    X = df.drop([df.columns[0], target_column], axis=1)  # Drop ID and target
    y = df[target_column]

    # Vérification du nombre de caractéristiques

    if X.shape[1] > 30:
        X = X.iloc[:, :30]

    elif X.shape[1] == 30:
        print("  Nombre de caractéristiques correct (30)")
    else:
        print(f"  Nombre inattendu de caractéristiques: {X.shape[1]}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Standardization (z-score)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

Data_Path = "../data/churn.csv"

def load_data(Data_Path):
    df = pd.read_csv(Data_Path)
    
    # Fix dataype issue
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop missing
    df.dropna(inplace=True)
    
    return df

def preprocess_data(df):
    # Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"]
    
    # Feature types
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    # Pipelines
    num_pipeline = Pipeline([
        ("scalar", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("encoding", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    # Feature Engineering
    df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsLongTerm"] = (df["tenure"] > 12).astype(int)
    
    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    df = load_data(Data_Path)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("✅ Data loaded and preprocessed successfully")
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)
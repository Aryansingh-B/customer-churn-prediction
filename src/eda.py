import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv(r"A:\DataScience_AI\Datasets\Churn.csv")

print(" First 5 rows:")
print(df.head())

print("\n Info:")
print(df.info())

print("\n Missing Values:")
print(df.isnull().sum())

# Convert TotalCharges to numeric (Important fix)
df["TotalChargers"] = pd.to_numeric(df["TotalChargers"], errors="coerce")

# Drop Missing
df.dropna(inplace = True)

# Distribution of target
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Numerical distribution
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in num_cols :
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.show()
    
# Correlation
corr = df[num_cols].corr()
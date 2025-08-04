import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# 🌟 Step 1: Select CSV files manually
Tk().withdraw()  # Hide the root window
print("🔍 Please select fraudTrain.csv")
train_path = askopenfilename(title="Select fraudTrain.csv")
print("🔍 Please select fraudTest.csv")
test_path = askopenfilename(title="Select fraudTest.csv")

# ✅ Step 2: Load datasets
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 🕒 Step 3: DateTime feature engineering
for df in [df_train, df_test]:
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df.drop('trans_date_trans_time', axis=1, inplace=True)

# 🔠 Step 4: Encode categorical columns
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    all_values = pd.concat([df_train[col].astype(str), df_test[col].astype(str)])
    le.fit(all_values)
    df_train[col] = le.transform(df_train[col].astype(str))
    df_test[col] = le.transform(df_test[col].astype(str))

# 🎯 Step 5: Split features and target
X = df_train.drop('is_fraud', axis=1)
y = df_train['is_fraud']
X_test = df_test.drop('is_fraud', axis=1)
y_test = df_test['is_fraud']

# 🧪 Step 6: Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 📊 Step 7: Scale numeric features
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 🌲 Step 8: Train Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
clf.fit(X_train, y_train)

# 📈 Step 9: Validation set evaluation
y_val_pred = clf.predict(X_val)
y_val_proba = clf.predict_proba(X_val)[:, 1]

print("\n=== Validation Set Evaluation ===")
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# 🧪 Step 10: Test set evaluation
y_test_pred = clf.predict(X_test)
y_test_proba = clf.predict_proba(X_test)[:, 1]

print("\n=== Test Set Evaluation ===")
print("Classification Report:\n", classification_report(y_test, y_test_pred))

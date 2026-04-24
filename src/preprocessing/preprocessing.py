import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('./data/artifacts', exist_ok=True)
os.makedirs('./data/preprocessed', exist_ok=True)

readings_df = pd.read_csv('data/raw/readings.csv')

# Remove unnecessary columns, drop duplicates and rows with missing values, and filter out rows with non-positive capacitance values.
df = readings_df.drop_duplicates()
df = df.dropna(subset=['capacitance_pf', 'category'])
df = df.drop(columns=['carried_out_at', 'day_of_week', 'hour_of_day', 'elapsed_minutes_since_first_reading', 'sample_label'])
df = df[df['capacitance_pf'] > 0]

# FOR CLASSIFICATION PREPROCESSING
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])

clf_features = ['capacitance_pf']

X_clf = df[clf_features].values  # raw, unscaled
y_clf = df['label'].values

scaler_clf = StandardScaler()
scaler_clf.fit(X_clf)  # fit only, don't transform yet — Pipeline will handle it

#Split RAW data — the Pipeline will scale internally during training
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_clf, y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf,
)

print(f'Classification split → train: {len(X_train_raw)}, test: {len(X_test_raw)}')

joblib.dump(scaler_clf, './data/artifacts/scaler_clf.pkl')

# FOR CLUSTERING PREPROCESSING
cluster_features = ['capacitance_pf']

X_cluster = df[cluster_features].values  # raw, unscaled

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

joblib.dump(scaler_cluster, './data/artifacts/scaler_cluster.pkl')

# Save RAW train/test splits for Pipeline training
np.save('./data/preprocessed/X_train.npy', X_train_raw)   # raw ← renamed, same variable name
np.save('./data/preprocessed/X_test.npy',  X_test_raw)    # raw ← renamed, same variable name
np.save('./data/preprocessed/y_train.npy', y_train)
np.save('./data/preprocessed/y_test.npy',  y_test)
np.save('./data/preprocessed/X_cluster.npy', X_cluster_scaled)

pd.DataFrame({
    'sample_id': df['sample_id'].values,
    'category':  df['category'].values,
    'label':     df['label'].values,
}).to_csv('./data/preprocessed/labels.csv', index=False)

print('\nPreprocessed files saved')
print(f'Label order (0, 1, 2 ...): {le.classes_.tolist()}')  # e.g. ['fresh', 'moderate', 'spoiled']
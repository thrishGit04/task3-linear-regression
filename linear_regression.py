# linear_regression.py (absolute paths)
import pandas as pd, numpy as np, os, json, joblib
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_CSV = r"/mnt/data/Housing.csv"
PROCESSED_CSV = r"/mnt/data/processed_Housing.csv"
OUT_DIR = r"/mnt/data/task3_outputs"
OUTPUTS_DIR = r"/mnt/data/task3_outputs/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
print("Loaded", INPUT_CSV, "shape", df.shape)

df = df.replace({'yes':1, 'no':0})
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
if obj_cols:
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
print("After encoding shape", df.shape)

df.to_csv(PROCESSED_CSV, index=False)
print("Saved processed CSV to", PROCESSED_CSV)

possible_targets = ['price','Price','SalePrice','house_value','HousePrice']
target = None
for t in possible_targets:
    if t in df.columns:
        target = t; break
if target is None:
    target = df.columns[-1]
print("Using target:", target)

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

scaler_path = os.path.join(OUTPUTS_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print("Saved scaler to", scaler_path)

model = LinearRegression()
model.fit(X_train, y_train)
model_path = os.path.join(OUTPUTS_DIR, 'model_linear_regression.joblib')
joblib.dump(model, model_path)
print("Saved model to", model_path)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
metrics = {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2)}
metrics_path = os.path.join(OUTPUTS_DIR, 'test_summary.json')
with open(metrics_path,'w') as fh: json.dump(metrics, fh, indent=2)
print("Saved metrics to", metrics_path)

coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_})
coef_path = os.path.join(OUTPUTS_DIR, 'coefficients.csv')
coef_df.to_csv(coef_path, index=False)
print("Saved coefficients to", coef_path)

numeric = df.select_dtypes(include=[np.number])
import matplotlib
matplotlib.use('Agg')
plt.figure(figsize=(10,8))
sns.heatmap(numeric.corr(), annot=True, fmt='.2f', cmap='coolwarm')
corr_path = os.path.join(OUT_DIR, 'correlation_matrix.png')
plt.tight_layout(); plt.savefig(corr_path); plt.close()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('Actual vs Predicted')
avp_path = os.path.join(OUT_DIR, 'actual_vs_predicted.png')
plt.tight_layout(); plt.savefig(avp_path); plt.close()

residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='dashed')
plt.xlabel('Predicted'); plt.ylabel('Residuals'); plt.title('Residuals vs Predicted')
res_path = os.path.join(OUT_DIR, 'residuals.png')
plt.tight_layout(); plt.savefig(res_path); plt.close()

plt.figure(figsize=(6,4))
sns.histplot(y, bins=30, kde=True)
hist_path = os.path.join(OUT_DIR, 'target_distribution.png')
plt.tight_layout(); plt.savefig(hist_path); plt.close()

print('Saved plots to', OUT_DIR)
print('\nALL DONE.')
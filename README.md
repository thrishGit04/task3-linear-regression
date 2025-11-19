**TASK 3 - LINEAR REGRESSION**

This repository contains Task 3 of my AIML Internship project.
The objective of this task is to build a Linear Regression model that predicts house prices using the Housing.csv dataset.

📁 Repository Structure
```
├── Housing.csv                         # Raw dataset
├── processed_Housing.csv               # Cleaned & encoded dataset
├── linear_regression.py                # Complete model training script
├── output
│   ├── model_linear_regression.joblib   # Trained Linear Regression model
│   ├── scaler.joblib                    # StandardScaler used for training
│   ├── test_summary.json                # MAE, MSE, RMSE, R² results
│   ├── coefficients.csv                 # Feature coefficients
│   ├── correlation_matrix.png           # Correlation heatmap
│   ├── actual_vs_predicted.png          # Actual vs predicted price plot
│   ├── residuals.png                    # Residuals vs predicted plot
│   ├── target_distribution.png          # Distribution of target variable
│   ├── run_stdout.txt                   # Execution logs (stdout)
│   └── run_stderr.txt                   # Execution logs (stderr)
└── README.md
```



**🧹 Data Preprocessing:**

```
Converted all "yes" / "no" values into 1 / 0
One-hot encoded remaining categorical (string) columns
Ensured the dataset contained only numeric features
```

**Split dataset into:**

```
X → Features
y → Target (price)
Scaled numerical columns using StandardScaler
Saved final cleaned file as processed_Housing.csv
```

🤖 Model Development

**Algorithm:** 
✔ Linear Regression

**Pipeline Steps:**

```
Load dataset
Clean & encode categorical features
Perform train-test split (80% train, 20% test)
Scale numeric columns
Train Linear Regression model
Make predictions on unseen test data
```

**Save:**

```
Model
Scaler
Coefficients
Evaluation metrics
Generate plots for analysis
```

**📈 Model Evaluation**
**Metrics saved in test_summary.json:**

```
MAE – Mean Absolute Error
MSE – Mean Squared Error
RMSE – Root Mean Squared Error
R² Score – How well model fits data
These metrics show how accurate the predictions are.
```

**📊 Generated Visualizations**

```
Stored in outputs
correlation_matrix.png
actual_vs_predicted.png
residuals.png
target_distribution.png
```

**These help understand:**

```
Feature relationships
Model fit quality
Error distribution
Target variable distribution
```

**🚀 How to Run This Project**

**Run with Python (Locally):** ``python linear_regression.py``

**Run in Google Colab:** ``Upload Housing.csv + linear_regression.py``

**Run:** ``!python linear_regression.py``

All outputs will be automatically created inside task3_outputs/.

**✨ Author**

Thrishool M S

AIML Internship — Task 3


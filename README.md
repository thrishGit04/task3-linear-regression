# ⭐ **Task 3 — Linear Regression (House Price Prediction)**

This repository contains **Task 3** of my AIML Internship project.
The goal of this task is to build a **simple & multiple Linear Regression model** to predict **house prices** using the Housing dataset.

The workflow includes preprocessing, model training, evaluation using regression metrics, and visual interpretation of model performance.

---

## 📁 **Repository Structure**

```
├── Housing.csv                     # Raw dataset (uploaded)
├── processed_Housing.csv           # Cleaned & preprocessed dataset
├── linear_regression.py            # Complete training script (single-run)
├── README.md                       # Documentation (this file)
└── output/
    ├── actual_vs_predicted.png         # Plot comparing true vs predicted prices
    ├── coefficients.csv                # Linear regression coefficients
    ├── correlation_matrix.png          # Correlation heatmap of features
    ├── empty                           # Placeholder auto-created by Colab
    ├── model_linear_regression.joblib  # Trained Linear Regression model
    ├── residuals.png                   # Residuals plot (errors vs predictions)
    ├── run_stderr.txt                  # Error logs (stderr)
    ├── run_stdout.txt                  # Execution logs (stdout)
    ├── scaler.joblib                   # StandardScaler used during training
    ├── target_distribution.png         # Distribution of the target variable
    └── test_summary.json               # MAE, MSE, RMSE, R² (evaluation metrics)
```

---

## 🎯 **Objective**

Implement and understand:

* **Simple Linear Regression**
* **Multiple Linear Regression**
* **Regression evaluation metrics**
* **Visualizing regression performance**

---

## 🧹 **Data Preprocessing Steps**

To prepare the data for regression:

1. Loaded the raw `Housing.csv` dataset.
2. Identified numerical and categorical features.
3. Handled missing values (median for numbers).
4. One-hot encoded categorical columns.
5. Scaled numerical columns using **StandardScaler**.
6. Saved the final preprocessed dataset as `processed_Housing.csv`.

---

## 🤖 **Model Training (linear_regression.py)**

The script:

* Reads `processed_Housing.csv`
* Splits data into **train/validation/test**
* Fits a **Linear Regression model**
* Evaluates using:

  * **MAE** (Mean Absolute Error)
  * **MSE** (Mean Squared Error)
  * **RMSE**
  * **R² Score**
* Saves:

  * trained model (`model_linear_regression.joblib`)
  * scaler (`scaler.joblib`)
  * coefficients (`coefficients.csv`)
  * summary (`test_summary.json`)

---

## 📊 **Generated Visualizations**

All stored inside `output/`:

### ✔ **Correlation Matrix**

Understanding relationship between features.
`correlation_matrix.png`

### ✔ **Actual vs Predicted Plot**

Shows how close predictions are to real house prices.
`actual_vs_predicted.png`

### ✔ **Residuals Plot**

Shows model errors. Ideal residuals cluster around zero.
`residuals.png`

### ✔ **Target Distribution**

Shows the distribution of house prices.
`target_distribution.png`

---

## 🧪 **Evaluation Metrics**

Stored in:

```
output/test_summary.json
```

Contains:

* **MAE**
* **MSE**
* **RMSE**
* **R² Score**

These metrics quantify model accuracy and error levels.

---

## 🚀 **How to Run the Project**

### **Option 1 — Google Colab (Recommended)**

Upload these files:

* `Housing.csv`
* `linear_regression.py`

Run:

```python
!python linear_regression.py
```

All outputs will be generated inside the `output/` folder.

---

### **Option 2 — Local Machine**

Install requirements:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

Run:

```bash
python linear_regression.py
```

---

## 📝 **Dataset**

**Housing Price Dataset**
A simple dataset typically used for regression learning tasks.

---

## ✨ **Author**

**Thrishool M S**

AIML Internship — *Task 3: Linear Regression*


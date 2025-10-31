# 🏠 Housing Price Predictor

A machine learning model for predicting Boston housing prices using regression techniques with comprehensive data preprocessing and feature engineering.

## ✨ Features

- **Robust Data Preprocessing**: Handles missing values, outliers, and feature scaling automatically
- **Multiple Algorithm Comparison**: Evaluates Linear Regression, Decision Trees, Random Forest, and ensemble methods
- **High Model Accuracy**: Achieves ~78-79% R² score with optimized Random Forest Regressor
- **Pre-trained Model**: Includes serialized model (`Price_Predictor.joblib`) for immediate inference
- **Production-Ready**: Complete pipeline from data ingestion to prediction with error handling

## 🚀 Quick Start/Installation

### Prerequisites
```bash
python 3.7+
```

### Clone Repository
```bash
git clone https://github.com/Nikshay-Jain/Housing-Price-Predictor.git
cd Housing-Price-Predictor
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

### Launch Jupyter Notebook
```bash
jupyter notebook
```

## 🛠️ Usage

### Training the Model

Open `Housing Price.ipynb` to train the model from scratch:

```python
# The notebook includes:
# 1. Data loading and exploration
# 2. Missing value imputation
# 3. Feature engineering and scaling
# 4. Model training with cross-validation
# 5. Performance evaluation and visualization
```

### Making Predictions

Use the pre-trained model with `Model Usage.ipynb`:

```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load('Price_Predictor.joblib')

# Prepare input features (13 attributes):
# CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
input_data = np.array([[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]])

# Predict housing price
predicted_price = model.predict(input_data)
print(f"Predicted Price: ${predicted_price[0] * 1000:.2f}")
```

### Input Features Description

The model requires 13 features from the Boston Housing Dataset:

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if bounds river; 0 otherwise) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to employment centers |
| RAD | Index of accessibility to radial highways |
| TAX | Property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents |
| LSTAT | % lower status of the population |

## 📊 Model Performance

**Random Forest Regressor Results** (from `Scores.txt.txt`):
- **Cross-Validation R² Score**: 0.7852 ± 0.0524
- **Training Method**: 10-fold cross-validation
- **Algorithm**: Random Forest with hyperparameter tuning
- **Dataset**: 506 samples with 13 features

## 📁 Repository Structure

```
Housing-Price-Predictor/
├── Housing Price.ipynb          # Main training pipeline
├── Model Usage.ipynb            # Inference and usage examples
├── Price_Predictor.joblib       # Serialized trained model
├── Scores.txt.txt               # Model evaluation metrics
├── housing.csv                  # Primary dataset (CSV format)
├── housing.data                 # Dataset (UCI ML format)
├── housing.names                # Dataset metadata and attribute descriptions
└── housing missing vals.csv     # Dataset variant with missing values for testing
```

## 🔍 Technical Details

**Tech Stack**:
- Python 3.x
- scikit-learn (ML algorithms and preprocessing)
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib & seaborn (visualization)
- joblib (model serialization)

**ML Pipeline**:
1. Data ingestion from CSV/UCI format
2. Exploratory data analysis with correlation heatmaps
3. Missing value imputation using median strategy
4. Feature scaling with StandardScaler
5. Model training with GridSearchCV for hyperparameter optimization
6. Cross-validation evaluation
7. Model persistence with joblib

## 📈 Dataset Information

**Source**: Boston Housing Dataset (UCI Machine Learning Repository)  
**Samples**: 506 observations  
**Features**: 13 continuous/discrete attributes  
**Target**: MEDV (Median value of homes in $1000s)  
**Use Case**: Regression problem for price prediction

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests with enhancements

## 📄 License

This project is open-source and available for educational and research purposes.

---

**Author**: [Nikshay Jain](https://github.com/Nikshay-Jain)  
**Last Updated**: November 2025

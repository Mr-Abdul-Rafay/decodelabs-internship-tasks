# 🌸 Project 2: Data Classification using KNN (Iris Dataset)

## 📌 Overview

This project implements a complete **Machine Learning classification pipeline** using the famous Iris dataset.
The goal is to accurately classify flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on their physical measurements.

The project demonstrates the full workflow of a real-world ML system, from data analysis to model deployment.

---

## 🚀 Objectives

* Perform Exploratory Data Analysis (EDA)
* Apply feature scaling and preprocessing
* Train a K-Nearest Neighbors (KNN) model
* Optimize the model by selecting the best K value
* Evaluate performance using multiple metrics
* Deploy a reusable prediction function

---

## 📊 Dataset Information

* Total Samples: **150**
* Features:

  * Sepal Length (cm)
  * Sepal Width (cm)
  * Petal Length (cm)
  * Petal Width (cm)
* Classes:

  * Setosa
  * Versicolor
  * Virginica

---

## ⚙️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Joblib (for model saving)

---

## 🔍 Project Workflow

### 1. Data Loading

* Loaded dataset using `sklearn.datasets`
* Converted to Pandas DataFrame
* Verified structure and class distribution

### 2. Exploratory Data Analysis (EDA)

* Checked missing values
* Correlation heatmap
* Pairplot visualization
* Feature distribution analysis
* Boxplots for outlier detection

### 3. Data Preprocessing

* Train-test split (80/20)
* Stratified sampling for balanced classes
* Feature scaling using `StandardScaler`

### 4. Model Training

* Implemented K-Nearest Neighbors (KNN)
* Tested K values from 1 to 30
* Selected optimal K based on accuracy & F1 score

### 5. Model Evaluation

* Accuracy Score
* F1 Score
* Confusion Matrix
* Classification Report

### 6. Deployment

* Created reusable prediction function
* Enabled batch predictions
* Interactive user input for real-time prediction
* Saved model using `.pkl` files

---

## 📈 Results

* ✅ Best K Value: *Optimized dynamically*
* ✅ Accuracy: **~95%+**
* ✅ F1 Score: **High and balanced across classes**
* ✅ Model shows excellent classification performance

---

## 📁 Project Structure

```
Project-2_Data_Classification/
│
├── main.py
├── iris_knn_model.pkl
├── scaler.pkl
├── correlation_heatmap.png
├── pairplot.png
├── feature_distributions.png
├── boxplots.png
├── confusion_matrix.png
├── k_value_optimization.png
└── README.md
```

---

## 💡 Key Insights

* Petal length and petal width are the most important features
* Setosa is linearly separable from other classes
* Versicolor and Virginica require ML for separation
* Proper feature scaling significantly improves model performance

---

## 🤖 Why KNN?

* Simple and intuitive algorithm
* No assumptions about data distribution
* Works well for small datasets
* Handles multi-class classification naturally

---

## 🔮 Future Improvements

* Implement cross-validation
* Compare with other models (Decision Tree, SVM)
* Apply feature selection techniques
* Build a web interface for predictions

---

## 👨‍💻 Author

**Abdul Rafay**
AI Engineer Intern

---

## 🎯 Conclusion

This project successfully demonstrates a complete machine learning pipeline, from data understanding to deployment.
It highlights strong practical knowledge of supervised learning and model evaluation techniques.

---

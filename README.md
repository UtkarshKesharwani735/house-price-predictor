# 🏠 House Price Prediction System

An end-to-end Machine Learning project that predicts residential house prices based on property attributes like area, location, number of bedrooms, and condition. Built using regression models and deployed as an interactive web application with Streamlit.

---

## 📌 Project Overview

This project follows the complete Machine Learning lifecycle — from data preprocessing and exploratory data analysis to model training, evaluation, and deployment — to predict house prices accurately based on user input.

**Objective:** Predict house prices based on factors such as area, location, number of rooms, age of property, and available amenities.

---

## 📂 Dataset

**File:** `House_Price_Prediction_Dataset.csv`

| Feature | Description | Type |
|---------|-------------|------|
| Area | Total built-up area (sq. ft.) | Numerical |
| Bedrooms | Number of bedrooms | Numerical |
| Bathrooms | Number of bathrooms | Numerical |
| Floors | Number of floors | Numerical |
| YearBuilt | Year of construction | Numerical |
| Location | Downtown / Urban / Suburban / Rural | Categorical |
| Condition | Excellent / Good / Fair / Poor | Categorical |
| Garage | Yes / No | Categorical |
| **Price** | Target variable (₹) | Numerical |

- **Total Records:** 2,000
- **Price Range:** ₹50,000 – ₹12,97,734

---

## 🧠 Machine Learning Concepts Used

- Data Cleaning
- Feature Engineering
- Regression Models
- Model Evaluation

---

## 🛠️ Project Development Methodology

| Phase | Description |
|-------|-------------|
| **Phase 1** | Problem Understanding — defining objectives and studying the dataset |
| **Phase 2** | Data Collection — loading the dataset |
| **Phase 3** | Data Preprocessing — handling missing values, duplicates, outliers, encoding |
| **Phase 4** | Exploratory Data Analysis — histograms, boxplots, scatter plots, heatmaps |
| **Phase 5** | Feature Engineering — creating and selecting meaningful features |
| **Phase 6** | Model Building — training Linear Regression, Random Forest, Gradient Boosting |
| **Phase 7** | Model Evaluation — MAE, MSE, RMSE, R² Score |
| **Phase 8** | Model Deployment — Streamlit web application |

---

## 🤖 Models Trained & Results

| Model | MAE (₹) | RMSE (₹) | R² Score |
|-------|---------|----------|----------|
| Linear Regression | 98,301 | 1,31,949 | 0.7113 |
| Random Forest | 28,088 | 37,118 | 0.9772 |
| **Gradient Boosting (Best)** | **23,630** | **32,149** | **0.9829** |

The **Gradient Boosting Regressor** was selected as the final model, achieving the highest accuracy.

---

## 📁 Project Structure

```
HOUSE-PRICE-PREDICTION/
│
├── House_Price_Prediction_Dataset.csv   # Dataset
├── House_Price_Prediction.ipynb         # Jupyter Notebook (all phases)
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
│
├── best_house_model.pkl                 # Saved trained model
├── scaler.pkl                           # Saved StandardScaler
├── label_encoders.pkl                   # Saved LabelEncoders
├── feature_names.pkl                    # Saved feature column order
│
└── README.md                            # Project documentation (this file)
```

---

## ⚙️ Installation & Setup

### 1. Clone or Download the Project
Place all project files in a single folder.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 3. Run the Jupyter Notebook (Optional — to retrain the model)
Open `House_Price_Prediction.ipynb` in Jupyter Notebook / VS Code and run all cells in order. This will regenerate the `.pkl` model files.

### 4. Run the Web Application
```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

---

## 🖥️ How to Use the App

1. Enter property details: Area, Bedrooms, Bathrooms, Floors, Year Built.
2. Select Location, Condition, and Garage availability.
3. Click **"Get Price Estimate"**.
4. View the predicted house price instantly.

---

## 📊 Key Visualizations

- Price distribution (histogram & boxplot)
- Feature correlation heatmap
- Scatter plots of features vs. price
- Feature importance (Random Forest)
- Model comparison charts (MAE, RMSE, R²)
- Actual vs. Predicted price plot

---

## 🔮 Future Scope

- Use larger, real-world datasets from multiple cities
- Add geospatial features (distance to schools, hospitals, transit)
- Try advanced models (XGBoost, LightGBM, Neural Networks)
- Hyperparameter tuning via Grid Search / Bayesian Optimization
- Deploy to cloud (Streamlit Community Cloud / AWS / Render)

---

## 🧰 Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib
- **Deployment:** Streamlit
- **Environment:** Jupyter Notebook / VS Code

---

## 📄 License

This project is created for academic and educational purposes.

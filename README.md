# 🏎️ F1 Race Position Predictor

This project builds a Random Forest model trained on real-world F1 race data to make intelligent predictions about race outcomes.

---

## 📌 Project Overview

Formula 1 is one of the most dynamic and thrilling motorsports in the world. However, predicting race results involves understanding many complex factors — driver skill, team performance, weather conditions, and more.

This project leverages machine learning to **predict F1 race finishing positions** based on historic race and driver data.  
It covers the complete pipeline from data collection to visualization of model insights.

---

## 📚 Data Collection and Exploratory Data Analysis (EDA)

- **Source:** Formula 1 World Championship (1950 - 2024)[https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020]
- **Steps Taken:**
  - Loaded datasets including driver details, race results, team performance, etc.
  - Performed data cleaning: handling missing values, standardizing column formats, and encoding categorical features.
  - Exploratory analysis included:
    - Checking correlations between features
    - Identifying important factors like starting grid, team performance, qualifying results
    - Visualizing distributions of race results

**EDA Insights:**  
- Drivers starting higher on the grid generally finish better.
- Teams like Mercedes, Red Bull show consistent dominance across seasons.
- Race conditions introduce variability but historical performance is a strong predictor.

---

## 🔥 Feature Engineering

- **Feature Selection:** Focused on features impacting race outcomes — grid position, driver ID, constructor ID (team), race ID, etc.
- **Encoding:** Applied Label Encoding for categorical variables like teams and drivers.
- **New Features:** 
  - Created aggregate statistics for drivers (e.g., average points, podium finishes)
  - Merged multiple tables to create a rich final dataset.
- **Final Target:** Predicted `positionOrder` (final race position) as the main label.

---

## 🧠 Machine Learning Model Application

- **Model Used:** `RandomForestClassifier`
- **Why Random Forest?**
  - Handles high-dimensional data well
  - Good for feature importance evaluation
  - Robust to overfitting with enough tuning
- **Hyperparameter Tuning:**  
  Used **GridSearchCV** to find the best parameters:
  - Number of trees (`n_estimators`)
  - Tree depth (`max_depth`)
  - Minimum samples split/leaf
  
- **Model Evaluation Metrics:**
  - **Accuracy**
  - **Confusion Matrix**
  - **Feature Importances**

**Saved Artifacts:**
- Trained Random Forest model (`.pkl` file)
- GridSearchCV results (`.csv` file)

---

## 🛠 Tech Stack Used

- **Python 3.12**
- **Libraries:**
  - `Pandas` — data manipulation
  - `NumPy` — numerical operations
  - `scikit-learn` — machine learning modeling
  - `Matplotlib` & `Seaborn` — data visualization
- **Tools:**
  - PyCharm IDE (project structuring)
  - Anaconda (environment management)

---

## 📈 Visualizations Created

- **Heatmaps:** Visualizing hyperparameter tuning results (accuracy vs parameters).
- **Confusion Matrix:** Analyzing model prediction errors.
- **Feature Importance Bar Charts:** Highlighting the most influential features in race outcomes.

---

## 📂 Project Structure

```
PythonProject1/
│
├── train_model.py       # Train and tune the Random Forest model
├── predict.py           # Load model and predict new driver race outcomes
├── visualization.py     # Generate heatmaps, confusion matrix, and bar charts
│
├── model/
│    ├── random_forest_model.pkl  # Saved best model
│    ├── gridsearch_results.csv   # Hyperparameter tuning results
│
├── cleaned_data.csv     # Final dataset after preprocessing
├── README.md            # (You are here!)
```

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

3. Make a prediction:
   ```bash
   python predict.py
   ```

4. Visualize model insights:
   ```bash
   python visualization.py
   ```

---

# ✨ Future Enhancements

- Implement advanced models like **XGBoost** for improved accuracy.
- Incorporate **weather conditions** and **pit stop strategies** as features.
- Build a **Streamlit** or **Gradio** app for interactive race prediction.
- Expand to multi-season analysis and real-time updates.

---

# 🏁 Conclusion

Machine learning brings a statistical edge to understanding and predicting the unpredictable world of Formula 1 racing.  
This project shows how careful data preparation, thoughtful feature engineering, and powerful models like Random Forest can make strong predictions in a high-variance sport like F1.



 🏎️ *"Turning raw F1 data into winning insights — one race at a time!"*



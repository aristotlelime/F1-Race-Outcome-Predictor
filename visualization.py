import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

best_rf = joblib.load('model/random_forest_model.pkl')

cv_results = pd.read_csv('model/gridsearch_results.csv')
X=pd.read_csv('model/X.csv')

# Load X_test and y_test
X_test = joblib.load('model/X_test.pkl')
X_train = joblib.load('model/X_train.pkl')
y_pred = joblib.load('model/y_pred.pkl')
y_test=joblib.load('model/y_test.pkl')

#Confusion Matrix
cm=confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix of Random Forest")


# Get feature importances from the trained model
importances = best_rf.feature_importances_

# Match them to column names
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort by importance
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# Plot as heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(feature_imp_df.set_index('Feature').T, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Importance'})
plt.title('Random Forest Feature Importances')
plt.yticks(rotation=0)

# Create pivot table for heatmap
pivot_table = cv_results.pivot_table(
    index='param_max_depth',
    columns='param_n_estimators',
    values='mean_test_score'
)


# Create a heatmap to visualize the performance
plt.figure(figsize=(10, 6))
pivot_table = cv_results.pivot_table(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Accuracy'}, linewidths=0.5)
plt.title('GridSearchCV - Random Forest Hyperparameter Tuning Results')
plt.ylabel('Max Depth')
plt.xlabel('Number of Estimators')

plt.show()
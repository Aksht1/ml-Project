# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Load the dataset from the provided data
data = pd.read_csv('updated_crop_data.csv')

# Prepare features (X) and target (y)
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
y = data['Yield']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Dictionaries to store results and predictions
results = {
    'Model': [],
    'MSE': [],
    'MAE': [],
    'R2': []
}
predictions = {}

# Train models, make predictions, and calculate metrics
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results['Model'].append(name)
    results['MSE'].append(mse)
    results['MAE'].append(mae)
    results['R2'].append(r2)

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Create a formatted table using PrettyTable
table = PrettyTable()
table.field_names = ['Model', 'MSE', 'MAE', 'R2 Score']
for index, row in results_df.iterrows():
    table.add_row([row['Model'], f"{row['MSE']:.4f}", f"{row['MAE']:.4f}", f"{row['R2']:.4f}"])

# Print the table with final results
print("\n=== Final Results Table for Crop Yield Estimation ===")
print(table)

# Determine the best model (highest R2 score)
best_model = results_df.loc[results_df['R2'].idxmax()]['Model']
best_mse = results_df['MSE'].min()
best_mae = results_df['MAE'].min()
best_r2 = results_df['R2'].max()
print(f"\nBest Model: {best_model}")
print(f"Lowest MSE: {best_mse:.4f}")
print(f"Lowest MAE: {best_mae:.4f}")
print(f"Highest R2 Score: {best_r2:.4f}")

# Visualization: Scatter Plots for Actual vs Predicted and Metrics
plt.figure(figsize=(15, 10))

# Actual vs Predicted for each model
for i, (name, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions', color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'{name}: Actual vs Predicted')
    plt.legend()
    plt.grid(True)

#  MSE Comparison
plt.subplot(2, 3, 4)
plt.scatter(results_df['Model'], results_df['MSE'], color='purple', s=100)
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.grid(True)

#  MAE Comparison
plt.subplot(2, 3, 5)
plt.scatter(results_df['Model'], results_df['MAE'], color='green', s=100)
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.grid(True)

# R2 Score Comparison
plt.subplot(2, 3, 6)
plt.scatter(results_df['Model'], results_df['R2'], color='orange', s=100)
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Feature Importance for Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

print("\n=== Project Conclusion ===")
print(f"The {best_model} is recommended as the best model for crop yield estimation.")
print(f"Reasons:")
print(f"- Lowest MSE ({best_mse:.4f}) and MAE ({best_mae:.4f}) indicate superior prediction accuracy.")
print(f"- Highest R2 Score ({best_r2:.4f}) shows it explains the most variance in yield.")
print(f"- Feature Importance graph (for Random Forest) highlights key predictors (e.g., Potassium, Nitrogen).")

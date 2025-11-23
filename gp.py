import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("updated_crop_data.csv")

# Define your feature columns and target (corrected to match dataset)
feature_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
target_column = 'Yield'

# Select features and target
X = df[feature_columns]
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Create metrics table with rounded values for readability
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R² Score'],
    'Value': [round(mae, 4), round(rmse, 4), round(r2, 4)]
})

print("\nModel Evaluation Metrics:")
print(metrics_df)

# Save metrics table as an image
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df.values,
                 colLabels=metrics_df.columns,
                 loc='center',
                 cellLoc='center',
                 colColours=['#f0f0f0', '#f0f0f0'])  # Light gray header
plt.title("Model Evaluation Metrics", pad=20)
plt.savefig("evaluation_metrics_table.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot actual vs predicted yield
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.show()
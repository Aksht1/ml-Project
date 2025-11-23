import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("updated_crop_data.csv")

# Define feature columns and target (using all relevant features from the dataset)
feature_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
target_column = 'Yield'

# Select features and target
X = df[feature_columns]
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models with tuned Random Forest
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=20,       # Reduced number of trees
        max_depth=5,          # Limited tree depth
        min_samples_leaf=5,   # Increased minimum samples per leaf
        random_state=42
    ),
    'SVR': SVR(kernel='rbf')
}

# Dictionary to store metrics for each model
metrics_data = {
    'Model': [],
    'MAE': [],
    'RMSE': [],
    'R² Score': []
}

# Train models, make predictions, and calculate metrics
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Store metrics
    metrics_data['Model'].append(model_name)
    metrics_data['MAE'].append(round(mae, 4))
    metrics_data['RMSE'].append(round(rmse, 4))
    metrics_data['R² Score'].append(round(r2, 4))

# Create a DataFrame for the comparison table
comparison_table = pd.DataFrame(metrics_data)

# Display the table in the console
print("\n=== Project Output: Model Comparison Table ===")
print(comparison_table)

# Create and save the table as an image
fig, ax = plt.subplots(figsize=(10, 3))  # Adjusted size for wider table
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=comparison_table.values,
                 colLabels=comparison_table.columns,
                 loc='center',
                 cellLoc='center',
                 colColours=['#f0f0f0'] * 4,  # Light gray header for all columns
                 bbox=[0, 0, 1, 1])  # Full figure bounding box
plt.title("Project Output: Comparison of Model Performance", pad=20)
plt.savefig("model_comparison_table.png", dpi=300, bbox_inches='tight')
plt.show()
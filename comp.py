from prettytable import PrettyTable

# Sample results (replace with your actual results from model evaluation)
results = {
    'Model': ['Linear Regression', 'Random Forest', 'SVR'],
    'MSE': [0.8234, 0.3245, 0.5678],
    'MAE': [0.6543, 0.4321, 0.5432],
    'R2': [0.6732, 0.8921, 0.7845]
}

# Create a formatted table using PrettyTable
table = PrettyTable()
table.field_names = ['Model', 'MSE', 'MAE', 'R2 Score']

# Add rows to the table
for i in range(len(results['Model'])):
    table.add_row([
        results['Model'][i],
        f"{results['MSE'][i]:.4f}",
        f"{results['MAE'][i]:.4f}",
        f"{results['R2'][i]:.4f}"
    ])

# Print the table
print("\n=== Final Results Table for Crop Yield Estimation ===")
print(table)
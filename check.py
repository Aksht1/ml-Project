import pandas as pd

# Load the dataset
df = pd.read_csv('Crop_Recommendation.csv')  # Replace with your file name

# Define yield values (tons/ha)
yield_dict = {
    # Original crops from your dataset
    'Rice': 4.7,
    'Maize': 5.9,
    'ChickPea': 1.1,
    'KidneyBeans': 1.5,
    'Cotton': 2.3,
    'Jute': 2.0,
    'Coffee': 0.7,
    'Coconut': 4.0,
    # Additional crops you requested
    'PigeonPeas': 0.9,
    'MothBeans': 0.4,
    'MungBean': 0.7,
    'Blackgram': 0.6,
    'Lentil': 1.3,
    'Pomegranate': 13.0,
    'Banana': 20.6,
    'Mango': 8.0,
    'Grapes': 9.8,
    'Watermelon': 25.0,
    'Muskmelon': 20.0,
    'Apple': 17.5,
    'Orange': 19.0,
    'Papaya': 13.5
}

# Add Yield column based on Crop
df['Yield'] = df['Crop'].map(yield_dict)

# Check for missing yields
if df['Yield'].isnull().any():
    missing_crops = df[df['Yield'].isnull()]['Crop'].unique()
    print("Warning: Some crops don’t have yield values assigned. Missing crops:", missing_crops)
else:
    print("All crops successfully assigned yield values.")

# Save the updated dataset
df.to_csv('updated_crop_data.csv', index=False)
print("Dataset updated and saved as 'updated_crop_data.csv'")
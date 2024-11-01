import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
def generate_user_data(n_samples=50):
    # Lists for generating random data
    names = [f"User_{i}" for i in range(1, n_samples + 1)]
    locations = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 
                'Berlin', 'Toronto', 'Mumbai', 'Singapore', 'Dubai']
    
    # Generate random attributes
    heights = np.random.normal(170, 10, n_samples)  # Mean 170cm, std 10cm
    weights = np.random.normal(70, 15, n_samples)   # Mean 70kg, std 15kg
    ages = np.random.randint(18, 70, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Name': names,
        'Location': [random.choice(locations) for _ in range(n_samples)],
        'Height': heights.round(1),
        'Weight': weights.round(1),
        'Age': ages
    })
    
    # Add fitness category based on BMI and age
    data['BMI'] = data['Weight'] / ((data['Height']/100) ** 2)
    data['FitnessCategory'] = data.apply(assign_fitness_category, axis=1)
    
    return data

def assign_fitness_category(row):
    bmi = row['BMI']
    age = row['Age']
    
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        if age < 30:
            return 'Fit Young'
        else:
            return 'Fit Mature'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Create the dataset
dataset = generate_user_data()

# Prepare data for KNN
# Convert categorical variables
location_encoded = pd.get_dummies(dataset['Location'], prefix='Location')
X = pd.concat([
    dataset[['Height', 'Weight', 'Age']],
    location_encoded
], axis=1)

y = dataset['FitnessCategory']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
k = 3  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Print results
print("\nOriginal Dataset Sample:")
print(dataset.head())

print("\nFeatures used for Classification:")
print(X.columns.tolist())

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to predict for new users
def predict_fitness_category(model, scaler, location_columns, height, weight, age, location):
    # Create a DataFrame for the new user
    new_user = pd.DataFrame([[height, weight, age]], columns=['Height', 'Weight', 'Age'])
    
    # Create location encoding
    location_data = pd.DataFrame(0, index=[0], columns=location_columns)
    location_col = f'Location_{location}'
    if location_col in location_columns:
        location_data[location_col] = 1
    
    # Combine features
    new_user = pd.concat([new_user, location_data], axis=1)
    
    # Scale the features
    new_user_scaled = scaler.transform(new_user)
    
    # Make prediction
    prediction = model.predict(new_user_scaled)
    return prediction[0]

# Example prediction for a new user
new_user_prediction = predict_fitness_category(
    knn, 
    scaler, 
    location_encoded.columns,
    height=175, 
    weight=70, 
    age=25, 
    location='London'
)
print("\nExample Prediction for new user:")
print(f"Height: 175cm, Weight: 70kg, Age: 25, Location: London")
print(f"Predicted Fitness Category: {new_user_prediction}")



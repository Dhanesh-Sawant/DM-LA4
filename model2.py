import pandas as pd
import numpy as np
from collections import defaultdict

# Create the dataset
data = {
    'Income': ['<30', '30-70', '30-70', '30-70', '30-70', '30-70', '>70', '>70', '<30', '30-70', '30-70', '30-70'],
    'Criminal_Record': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'],
    'Experience': ['1-5', '1', '1', '1-5', '>5', '1-5', '>5', '>5', '1-5', '1-5', '1-5', '>5'],
    'Loan_Approved': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}  # P(Class)
        self.feature_probs = {}  # P(Feature|Class)
        self.classes = None
        
    def calculate_probability(self, feature_counts, total_count, unique_values):
        # Add Laplace smoothing
        return {value: (count + 1) / (total_count + len(unique_values)) 
                for value, count in feature_counts.items()}
    
    def fit(self, X, y):
        self.classes = y.unique()
        total_samples = len(y)
        
        # Calculate class probabilities P(Class)
        class_counts = y.value_counts()
        self.class_probs = {cls: count/total_samples for cls, count in class_counts.items()}
        
        # Calculate conditional probabilities P(Feature|Class)
        self.feature_probs = defaultdict(dict)
        
        for feature in X.columns:
            unique_values = X[feature].unique()
            
            for cls in self.classes:
                # Get samples for this class
                class_data = X[y == cls]
                feature_counts = class_data[feature].value_counts().to_dict()
                
                # Calculate P(Feature|Class) with Laplace smoothing
                self.feature_probs[feature][cls] = self.calculate_probability(
                    feature_counts, 
                    len(class_data), 
                    unique_values
                )
                
                # Add smoothing for unseen values
                for value in unique_values:
                    if value not in self.feature_probs[feature][cls]:
                        self.feature_probs[feature][cls][value] = 1 / (len(class_data) + len(unique_values))
    
    def predict_proba(self, X):
        probabilities = {}
        
        for cls in self.classes:
            # Start with P(Class)
            prob = np.log(self.class_probs[cls])
            
            # Multiply by P(Feature|Class) for each feature
            for feature, value in X.items():
                if value in self.feature_probs[feature][cls]:
                    prob += np.log(self.feature_probs[feature][cls][value])
                else:
                    # Handle unseen values
                    prob += np.log(1 / (len(self.feature_probs[feature][cls]) + 1))
            
            probabilities[cls] = prob
        
        # Convert log probabilities to regular probabilities
        max_prob = max(probabilities.values())
        exp_probs = {k: np.exp(v - max_prob) for k, v in probabilities.items()}
        total = sum(exp_probs.values())
        return {k: v/total for k, v in exp_probs.items()}
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return max(probs.items(), key=lambda x: x[1])[0]

# Train the model
X = df[['Income', 'Criminal_Record', 'Experience']]
y = df['Loan_Approved']

nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X, y)

# Function to print probabilities in a readable format
def print_probabilities(probs):
    for cls, prob in probs.items():
        print(f"Probability of {cls}: {prob:.2%}")

# Make prediction for the given case
new_case = {
    'Income': '30-70',
    'Criminal_Record': 'Yes',
    'Experience': '>5'
}

# Calculate probabilities
probabilities = nb_classifier.predict_proba(new_case)
prediction = nb_classifier.predict(new_case)

print("\nInput Data Summary:")
print(df)

print("\nProbability Analysis for New Case:")
print("Case details:", new_case)
print("\nCalculated Probabilities:")
print_probabilities(probabilities)
print(f"\nFinal Prediction: {prediction}")

# Print the learned probabilities for transparency
print("\nLearned Model Parameters:")
print("\nClass Probabilities P(Class):")
for cls, prob in nb_classifier.class_probs.items():
    print(f"{cls}: {prob:.2f}")

print("\nConditional Probabilities P(Feature|Class):")
for feature in nb_classifier.feature_probs:
    print(f"\n{feature}:")
    for cls in nb_classifier.classes:
        print(f"{cls}:", nb_classifier.feature_probs[feature][cls])
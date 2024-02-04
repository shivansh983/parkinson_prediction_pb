import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # For model persistence

# Load the dataset
df = pd.read_csv('parkinsons.csv')  # Replace 'parkinsons.csv' with the actual path to your dataset

# Prepare data for modeling
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

# Splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training (Support Vector Machine)
model = SVC(kernel="linear")
model.fit(X_train_scaled, Y_train)

# Predictions on training and test data
Y_train_pred = model.predict(X_train_scaled)
Y_test_pred = model.predict(X_test_scaled)

# Compute accuracy scores
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

# Print accuracy scores
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model as a .pkl file
joblib.dump(model, 'parkinsons_model.pkl')

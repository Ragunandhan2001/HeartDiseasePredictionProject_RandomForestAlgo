# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib 

# df = pd.read_csv("heart_disease_data.csv")

# print("missing data...",df.isnull().sum())

# X = df.drop("target",axis=1)
# y = df["target"]

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# model = RandomForestClassifier()

# model.fit(X_train,y_train)

# accuracy = model.score(y_test,y_test)
# print("model accuracy :",accuracy)

# joblib.dump(model,"heart_model.pkl")
# print("Heart Diseases prediction Model saved!")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("heart_disease_data.csv")

# 🧹 Data Processing using Pandas
print("Missing values:\n", df.isnull().sum())

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Split data (NEW CONCEPT)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = RandomForestClassifier()

# Train
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "heart_model.pkl")
print("Heart model saved!")

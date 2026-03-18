import pandas as pd

data = {
    "Study_Hours": [2, 4, 6, 8, 1, 3, 5, 7, 9, 10],
    "Sleep_Hours": [7, 6, 5, 6, 8, 7, 6, 5, 4, 3],
    "Pass": [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = df[["Study_Hours", "Sleep_Hours"]]
y = df["Pass"]

# Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

print("\nTraining Data:")
print(X_train, y_train)

print("\nTesting Data:")
print(X_test, y_test)
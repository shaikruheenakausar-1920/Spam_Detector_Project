import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load data
df = pd.read_csv("Advertising.csv")

# Drop the unnamed index column if present
if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
    df = df.drop(df.columns[0], axis=1)

# 2. Features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Check accuracy
score = model.score(X_test, y_test)
print(f"Model R2 score: {score:.3f}")

# 6. Save model
pickle.dump(model, open("sales_model.pkl", "wb"))

print("Model saved successfully!")

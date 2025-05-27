import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('BIKE DETAILS.csv')

# Drop rows with missing target or essential features
df.dropna(subset=['selling_price', 'km_driven'], inplace=True)

# Clean and convert km_driven
df['km_driven'] = df['km_driven'].astype(str).str.replace(',', '').astype(float)

# Fill missing ex_showroom_price with mean
df['ex_showroom_price'] = df['ex_showroom_price'].fillna(df['ex_showroom_price'].mean())

# Extract bike brand name
df['bike_name'] = df['name'].str.split().str[0].str.lower()

# Map 'owner' values properly
owner_map = {
    '1st owner': 1,
    '2nd owner': 2,
    '3rd owner': 3,
    '4th owner': 4,
    '5th owner and above': 5,
    'Test Drive': 0
}
df['owner'] = df['owner'].map(owner_map)

# Drop any rows with unknown owner values after mapping
df.dropna(subset=['owner'], inplace=True)

# One-hot encode seller_type
df = pd.get_dummies(df, columns=['seller_type'], drop_first=True)

# Label encode bike_name
bike_le = LabelEncoder()
df['bike_name'] = bike_le.fit_transform(df['bike_name'])

# Final feature selection
features = ['year', 'km_driven', 'ex_showroom_price', 'owner', 'bike_name'] + \
           [col for col in df.columns if col.startswith('seller_type_')]
X = df[features]
y = df['selling_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print evaluation
print(f"Training R² score: {train_score:.4f}")
print(f"Testing R² score: {test_score:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save model and encoders
joblib.dump(model, 'bike_price_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(bike_le, 'bike_label_encoder.joblib')

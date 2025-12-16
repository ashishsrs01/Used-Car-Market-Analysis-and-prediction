import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv("C:/Project/Used-Car-Market-Analysis-and-prediction/Car details v3.csv")

def preprocess_data(df):
    """Preprocess the data by handling missing values and converting data types."""
    # Check for missing values
    if df.isnull().sum().any():
        df = df.dropna()  # Drop rows with missing values for simplicity

    # Convert 'Year' to age of the car
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Year']
    df.drop(columns=['Year'], inplace=True)

    # Convert categorical columns to category dtype
    categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df

def train_model(df):
    """Train a Random Forest model to predict car prices."""
    # Define features and target variable
    X = df.drop(columns=['Price', 'Car_Name'])
    y = df['Price']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Model Evaluation:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")

    return model
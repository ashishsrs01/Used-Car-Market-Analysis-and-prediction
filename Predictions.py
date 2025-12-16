import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data by handling missing values and converting data types."""
    # Rename columns to match the code's expectations
    df.rename(columns={
        'name': 'Car_Name',
        'year': 'Year',
        'selling_price': 'Price',
        'km_driven': 'KM_Driven',
        'fuel': 'Fuel_Type',
        'seller_type': 'Seller_Type',
        'transmission': 'Transmission',
        'owner': 'Owner'
    }, inplace=True)

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
    X = pd.get_dummies(X, drop_first=False)
    
    # Store the feature names for later use in prediction
    train_model.feature_columns = X.columns.tolist()

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

    print(f"Model Evaluation:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}")

    return model

def predict_price(model, input_data):
    """Predict the price of a car given input features."""
    input_df = pd.DataFrame([input_data])
    
    # Convert categorical columns to match training data types
    categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')
    
    input_df = pd.get_dummies(input_df, drop_first=False)

    # Align input data with training data columns
    for col in train_model.feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[train_model.feature_columns]

    predicted_price = model.predict(input_df)
    return predicted_price[0]
    
def main():
    # Load and preprocess data
    df = load_data("C:/Project/Used-Car-Market-Analysis-and-prediction/Car details v3.csv")
    df = preprocess_data(df)

    # Train the model
    model = train_model(df)

    # Example prediction
    example_input = {
        'Age': 5,
        'KM_Driven': 50000,
        'Fuel_Type': 'Petrol',
        'Seller_Type': 'Individual',
        'Transmission': 'Manual',
        'Owner': 'First Owner'
    }
    predicted_price = predict_price(model, example_input)
    print(f"Predicted Price: {predicted_price:.2f}")

if __name__ == "__main__":
    main()
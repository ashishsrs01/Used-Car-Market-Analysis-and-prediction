import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def analyze_data(df):
    """Perform exploratory data analysis on the DataFrame."""
    print("Data Summary:")
    print(df.describe(include='all'))

    # Correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Distribution of Price
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Price'], bins=30, kde=True)
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

    # Price vs Age
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Age', y='Price', data=df)
    plt.title('Price vs Age of Car')
    plt.xlabel('Age (years)')
    plt.ylabel('Price')
    plt.show()
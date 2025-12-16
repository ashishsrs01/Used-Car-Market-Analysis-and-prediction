import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data by handling missing values and converting data types."""
    # Check for missing values
    if df.isnull().sum().any():
        df = df.dropna()  # Drop rows with missing values for simplicity

    # Convert 'year' to age of the car
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['year']
    df.drop(columns=['year'], inplace=True)

    # Convert categorical columns to category dtype
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df

def analyze_data(df):
    """Perform exploratory data analysis on the DataFrame."""
    print("Data Summary:")
    print(df.describe(include='all'))

    # Correlation matrix (only numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Distribution of selling_price
    plt.figure(figsize=(8, 5))
    sns.histplot(df['selling_price'], bins=30, kde=True)
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

    # Price vs Age
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Age', y='selling_price', data=df)
    plt.title('Price vs Age of Car')
    plt.xlabel('Age (years)')
    plt.ylabel('Price')
    plt.show()

    # Price by Fuel Type
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='fuel', y='selling_price', data=df)
    plt.title('Price by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Price') 
    plt.show()

    # Price by Transmission
    plt.figure(figsize=(8, 5))              
    sns.boxplot(x='transmission', y='selling_price', data=df)
    plt.title('Price by Transmission Type')
    plt.xlabel('Transmission')
    plt.ylabel('Price')
    plt.show()

    # Price by Seller Type
    plt.figure(figsize=(8, 5))      
    sns.boxplot(x='seller_type', y='selling_price', data=df)
    plt.title('Price by Seller Type')
    plt.xlabel('Seller Type')
    plt.ylabel('Price')
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    data = load_data("C:/Project/Used-Car-Market-Analysis-and-prediction/Car details v3.csv")
    preprocessed_data = preprocess_data(data)

    # Analyze data
    analyze_data(preprocessed_data)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import optuna
import pickle
from collections import Counter

from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# To create boxplots for each column in a DataFrame and display them in a 3x3 grid
def graficar_box(df_box):
    """
    Creates boxplots for each column in the DataFrame (except 'smoking') and plots them in a 3x3 grid.

    Parameters:
    df_box (DataFrame): DataFrame containing the data to plot. It should include a 'smoking' column which is excluded from the plots.

    Returns:
    None
    """
    columns = [col for col in df_box.columns if col != 'smoking']
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Calcula el número de filas necesarias

    fig, axs = plt.subplots(num_rows, 3, figsize=(20, 4 * num_rows))  # Ajusta el tamaño de la figura

    for i, column in enumerate(columns):
        row = i // 3
        col = i % 3
        sns.boxplot(data=df_box[column], orient="h", ax=axs[row, col])
        axs[row, col].set_title(f"Boxplot for {column}")
        axs[row, col].set_xlabel("Value")
    
    # Elimina los ejes sobrantes si hay menos de 9 gráficos
    for j in range(i + 1, num_rows * 3):
        row = j // 3
        col = j % 3
        fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.show()


# # To create distplots for each column in a DataFrame and display them in a 3x3 grid
def graficar_dist(df_dist):
    """
    Creates histograms with distribution lines for each column in the DataFrame (except 'smoking') and plots them in a 3x3 grid.

    Parameters:
    df_dist (DataFrame): DataFrame containing the data to plot. It should include a 'smoking' column to be used as a hue in the plots.

    Returns:
    None
    """
    columns = [col for col in df_dist.columns if col != 'smoking']
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Calcula el número de filas necesarias

    fig, axs = plt.subplots(num_rows, 3, figsize=(20, 4 * num_rows))  # Ajusta el tamaño de la figura
    axs = axs.flatten()  # Aplanar la matriz de ejes para un acceso fácil

    for i, column in enumerate(columns):
        lower_percentile = df_dist[column].quantile(0.05)
        upper_percentile = df_dist[column].quantile(0.95)

        plt.style.use("fivethirtyeight")
        sns.set_style("whitegrid")
        sns.histplot(data=df_dist, x=column, hue='smoking', kde=True, palette='Set1', element='step', ax=axs[i])

        axs[i].set_xlim(lower_percentile, upper_percentile)
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Histogram with Distribution Line for {column}')

    # Elimina los ejes sobrantes si hay menos de 9 gráficos
    for j in range(i + 1, num_rows * 3):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()


# To create histplots for each column in a DataFrame and display them in a 3x3 grid
def graficar_hist(df_hist, columns_categorical):
    """
    Creates bar plots for each categorical column in the DataFrame and plots them in a 3x3 grid.

    Parameters:
    df_hist (DataFrame): DataFrame containing the data to plot.
    columns_categorical (list of str): List of categorical columns to plot.

    Returns:
    None
    """
    num_columns = len(columns_categorical)
    num_rows = (num_columns + 2) // 3  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, 3, figsize=(20, 4 * num_rows))  # Adjust the figure size
    axs = axs.flatten()  # Flatten the axes array for easy access

    for i, column in enumerate(columns_categorical):
        sns.countplot(data=df_hist, x=column, hue='smoking', palette='Set1', ax=axs[i])

        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Count')
        axs[i].set_title(f'Bar Plot for {column}')

    # Remove any extra axes if there are less than 9 plots
    for j in range(i + 1, num_rows * 3):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()


# To label encode 'gender' and 'tartar' columns
def label_encode(df):
    """
    Encodes categorical columns in the DataFrame to numerical values.

    Parameters:
    df (DataFrame): DataFrame containing the data to encode. It should include 'gender' and 'tartar' columns.

    Returns:
    None
    """
    df['gender'] = df['gender'].replace({'M': 1, 'F': 0})
    # Encode 'tartar' so it can be used by our model
    df['tartar'] = df['tartar'].replace({'Y': 1, 'N': 0})


# To transform positively skewed distributions
def transformar_asimetricas(df):
    """
    Applies logarithmic transformation to specified columns with positively skewed distributions.

    Parameters:
    df (DataFrame): DataFrame containing the data to transform. It should include the specified columns.

    Returns:
    None
    """
    # List of columns to transform
    columns_to_transform = ['fasting blood sugar', 'triglyceride', 'HDL', 'AST', 'ALT', 'Gtp', 'Urine protein']

    # Define the logarithmic transformer
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    # Apply the logarithmic transformation to the specified columns
    df[columns_to_transform] = log_transformer.transform(df[columns_to_transform])


# To normalize columns of a dataframe
def normalizar(df, columns_to_normalize):
    """
    Normalizes specified columns in the DataFrame to a range between 0 and 1.

    Parameters:
    df (DataFrame): DataFrame containing the data to normalize. It should include the specified columns.
    columns_to_normalize (list of str): List of column names to normalize.

    Returns:
    None
    """
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Apply the scaler to the specified columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


# Applies one-hot encoding to specified columns in the DataFrame
def encode_df(df_to_encode, columns_to_encode):
    """
    Encodes specified columns in the DataFrame using one-hot encoding and appends the encoded columns to the original DataFrame.

    Parameters:
    df_to_encode (DataFrame): DataFrame containing the data to encode. It should include the specified columns.
    columns_to_encode (list of str): List of column names to one-hot encode.

    Returns:
    DataFrame: The DataFrame with the specified columns one-hot encoded and added to the original DataFrame, with the original columns removed.
    """
    # Load the OneHotEncoder model
    ohe = OneHotEncoder(sparse_output=False).fit(df_to_encode[columns_to_encode])
    # Transform the columns to encode
    encoded = ohe.transform(df_to_encode[columns_to_encode])
    # Save the encoded data in a DataFrame
    encoded_df = pd.DataFrame(columns=ohe.get_feature_names_out(), data=encoded, index=df_to_encode.index)
    # Concatenate the original DataFrame with the encoded DataFrame
    df_to_encode = pd.concat([df_to_encode, encoded_df], axis='columns')
    # Drop the original columns that were encoded
    df_to_encode.drop(columns=columns_to_encode, inplace=True)
    return df_to_encode

# To apply train-test split to a dataframe
def split(df_split):
    """
    Splits the DataFrame into training and testing sets for features and target variable.

    Parameters:
    df_split (DataFrame): DataFrame containing the features and the target variable ('smoking').

    Returns:
    tuple: A tuple containing four DataFrames: X_train, X_test, y_train, and y_test.
           - X_train: Training features
           - X_test: Testing features
           - y_train: Training target variable
           - y_test: Testing target variable
    """
    x = df_split.drop('smoking', axis=1).copy()
    y = df_split.smoking.copy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# To plot the feature importance of a trained model
def plot_feature_importance(X_train, clf):
    """
    Plots the feature importance of a trained model.

    Parameters:
    X_train (DataFrame): DataFrame containing the training features.
    clf (model): Trained model with a `feature_importances_` attribute.

    Returns:
    None
    """
    # Create a DataFrame to store feature names and their importance scores
    fi = pd.DataFrame(columns=['FEATURE', 'IMPORTANCE'])
    fi['FEATURE'] = X_train.columns
    fi['IMPORTANCE'] = clf.feature_importances_
    
    # Sort the DataFrame by importance in descending order
    fi = fi.sort_values('IMPORTANCE', ascending=False)
    
    # Plot the feature importance
    plt.figure(figsize=(5, 15))
    sns.barplot(y=fi.FEATURE, x=fi.IMPORTANCE)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()


# To optimize hyperparameters for a RandomForestClassifier using Optuna
def objective_rfc(trial):
    """
    Objective function for hyperparameter optimization using Optuna for a RandomForestClassifier.

    Parameters:
    trial (optuna.Trial): An Optuna trial object used to suggest hyperparameter values.

    Returns:
    float: The mean cross-validation accuracy of the RandomForestClassifier with suggested hyperparameters.
    """
    # Suggest values for hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    
    # Create the classifier with suggested hyperparameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    # Evaluate the model using cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    accuracy = scores.mean()
    
    return accuracy


# Objective function for hyperparameter optimization using Optuna for a Logistic Regression model# Objective function for hyperparameter optimization using Optuna for a Logistic Regression model
def objective_lr(trial, X_train, y_train):
    """
    Objective function for hyperparameter optimization using Optuna for a Logistic Regression model.

    Parameters:
    trial (optuna.Trial): An Optuna trial object used to suggest hyperparameter values.
    X_train (DataFrame): The training features.
    y_train (Series): The training target variable.

    Returns:
    float: The mean cross-validation accuracy of the Logistic Regression model with suggested hyperparameters.
    """
    # Suggest values for hyperparameters
    C = trial.suggest_loguniform('C', 1e-6, 1e2)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) 

    # Create the classifier with suggested hyperparameters
    clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter, l1_ratio=l1_ratio, random_state=42)
    
    # Evaluate the model using cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    accuracy = scores.mean()
    
    return accuracy
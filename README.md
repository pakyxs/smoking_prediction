# Smoking Prediction based on clinical information

## Overview

This project aims to predict whether a patient is a smoker based on various clinical information. The model uses machine learning techniques to analyze clinical data and make predictions, which can assist healthcare professionals in identifying smoking habits among patients.

## Features

- **Data Preprocessing**: Cleaning and transforming raw clinical data for analysis.
- **Feature Engineering**: Creating meaningful features from clinical data to improve model accuracy.
- **Model Training**: Implementing and training various machine learning models to predict smoking status.
- **Evaluation**: Assessing model performance using accuracy.
- **Deployment**: Deploying the model using a web interface or API.

## Dataset

The dataset used for this project contains clinical information of patients, including but not limited to:

- Age
- Gender
- BMI
- Blood pressure
- Cholesterol levels
- Heart rate
- Medical history
- Smoking status (target variable)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/pakyxs/smoking-prediction.git
    ```
   
2. **Navigate to the project directory**:
    ```bash
    cd smoking-prediction
    ```

3. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Docker

To run the project using Docker:

1. **Build the Docker image:**

   ```bash
   docker build -t smoking-prediction .
   ```

2. **Run the Docker container:**

   ```bash
   docker run -v $(pwd)/data:/app/data smoking-prediction
   ```

   This command mounts the `data` directory from your host into the container, allowing the container to access your dataset.

## Project Structure

- `data/`: Contains raw and processed datasets.
- `functions/`: Contains custom functions used in this project.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and model experimentation.
- `src/`: Contains the core Python scripts for data preprocessing, model training, and evaluation.
- `models/`: Saved machine learning models.
- `requirements.txt`: List of required Python packages.

## Model Details

The project explores various machine learning algorithms, including:

- Logistic Regression
- Random Forest
- XGBoost

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to reach out:


- **GitHub**: [pakyxs](https://github.com/pakyxs)



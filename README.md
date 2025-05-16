# House-price-predictor

House Price Prediction

This project is a Python-based house price prediction model using linear regression. It leverages the scikit-learn library to predict house prices based on features like square footage, number of bedrooms, and bathrooms from a dataset (train.csv). The model is trained, evaluated, and can make predictions for new houses.

Features





Data Preprocessing: Handles missing values and creates a composite feature for bathrooms (TotalBath).



Model Training: Uses linear regression to predict house prices based on GrLivArea, BedroomAbvGr, and TotalBath.



Evaluation: Computes Root Mean Squared Error (RMSE) and R-squared score to assess model performance.



Prediction: Provides price predictions for new houses with specified features.



Simple and Extensible: Easy to modify for additional features or different models.

Prerequisites





Python 3.8 or higher



A dataset (train.csv) with columns: GrLivArea, BedroomAbvGr, FullBath, HalfBath, and SalePrice. Example: Kaggle House Prices Dataset

Installation





Clone the repository:

git clone <repository-url>
cd house-price-prediction



Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install the required dependencies:

pip install -r requirements.txt



Place the train.csv dataset in the specified directory (e.g., D:\house price prediction\train.csv) or update the file path in the script.

Usage





Ensure the train.csv dataset is accessible.



Run the script:

python house_price_prediction.py



The script will:





Load and preprocess the dataset.



Train a linear regression model.



Output model coefficients, intercept, and performance metrics (RMSE, R-squared).



Predict the price for an example house (2000 sqft, 3 bedrooms, 2.5 bathrooms).



View the results in the console.

Example Output

Model Coefficients:
GrLivArea: 107.23
BedroomAbvGr: -12345.67
TotalBath: 23456.89
Intercept: 45000.12

Model Performance:
Root Mean Squared Error: 40000.50
R-squared Score: 0.75

Predicted price for a 2000 sqft house with 3 bedrooms and 2.5 bathrooms: $256,789.45

Project Structure





house_price_prediction.py: Main script containing the prediction logic.



requirements.txt: List of Python dependencies.



train.csv: Dataset file (not included; must be provided by the user).

Dependencies

See requirements.txt for the full list of dependencies, including:





pandas: For data loading and manipulation.



numpy: For numerical computations.



scikit-learn: For linear regression and model evaluation.

Notes





Ensure the train.csv file path in the script matches your local setup.



The model assumes linear relationships between features and price; consider exploring other models (e.g., Random Forest) for better performance.



Missing values are filled with feature means; adjust preprocessing as needed for your dataset.



The example prediction can be modified by changing the example_house DataFrame.

Troubleshooting





File Not Found Error: Verify the train.csv path is correct and the file exists.



Dependency Issues: Ensure all packages in requirements.txt are installed (pip install -r requirements.txt).



Data Issues: Check that the dataset has the required columns (GrLivArea, BedroomAbvGr, FullBath, HalfBath, SalePrice).

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def add_new_entry(df):
    new_entry = {
        'SL.NO': len(df) + 1,
        'COMPANY NAME': input("Enter Company Name: "),
        'LOAN AMOUNT': float(input("Enter Loan Amount: ")),
        'LIQUIDITY': float(input("Enter Liquidity: ")),
        'LOAN AMOUNT REFUNDED': float(input("Enter Loan Amount Refunded: ")),
        'LOCATION OF COMPANY': input("Enter Location of Company: "),
        'COMPANY CURRENT FUNDS': float(input("Enter Company Current Funds: ")),
        'FREQUENCY OF TAKING LOANS': int(input("Enter Frequency of Taking Loans: ")),
        'LOAN ID': input("Enter Loan ID: "),
        'LOAN AMOUNT TERM': float(input("Enter Loan Amount Term: ")),
        'CREDIT HISTORY': float(input("Enter Credit History: ")),
        'LOAN REQUEST DATE': input("Enter Loan Request Date (comma-separated if multiple): ")
    }

    new_entry_df = pd.DataFrame([new_entry])
    df = pd.concat([df, new_entry_df], ignore_index=True)
    return df


# Load the dataset
try:
    df = pd.read_excel('C:\\Users\\frank\\OneDrive\\Desktop\\BOLT\\dataset1.xlsx')
except FileNotFoundError as e:
    print(f"Error loading the file: {e}")
    raise

# Add new data entry
df = add_new_entry(df)

# Feature Engineering: Creating 'LOAN REQUEST COUNT'
if 'LOAN REQUEST DATE' in df.columns:
    df['LOAN REQUEST COUNT'] = df['LOAN REQUEST DATE'].apply(lambda x: len(str(x).split(',')))
else:
    print("'LOAN REQUEST DATE' column not found in the dataset. Ensure the dataset is correct.")

# Verify 'LOAN REQUEST COUNT' creation
if 'LOAN REQUEST COUNT' not in df.columns:
    print("Failed to create 'LOAN REQUEST COUNT'. Please check the data and creation logic.")
    raise Exception("Missing 'LOAN REQUEST COUNT' column.")

# Prepare features and target variable
features = ['FREQUENCY OF TAKING LOANS', 'CREDIT HISTORY', 'LOAN REQUEST COUNT']
X = df[features]
y = df['LOAN AMOUNT']

# Define preprocessing steps
numeric_features = features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Apply PCA after scaling and polynomial features
pca_transformer = PCA(n_components=0.95)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

# Define the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca_transformer),
    ('model', model)
])

# Setup hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X, y)

# Model Evaluation
mse = mean_squared_error(y, grid_search.predict(X))
r2 = r2_score(y, grid_search.predict(X))

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Predict and find the company with the highest predicted loan amount
df['PREDICTED LOAN AMOUNT'] = grid_search.predict(X)
highest_loan_company = df.loc[df['PREDICTED LOAN AMOUNT'].idxmax()]

print("Company Name likely to ask for the highest Loan:", highest_loan_company['COMPANY NAME'])
print("Highest Predicted Loan Amount:", highest_loan_company['PREDICTED LOAN AMOUNT'])

# Calculate financial capacity
df['FINANCIAL CAPACITY'] = df['LIQUIDITY'] + df['COMPANY CURRENT FUNDS']

# Identify potential lenders
potential_lenders = df[df['FINANCIAL CAPACITY'] >= highest_loan_company['PREDICTED LOAN AMOUNT']]
potential_lenders['CAPACITY DIFFERENCE'] = potential_lenders['FINANCIAL CAPACITY'] - highest_loan_company['PREDICTED LOAN AMOUNT']
best_lender = potential_lenders.loc[potential_lenders['CAPACITY DIFFERENCE'].idxmin()]

print("Best Lender Company Name:", best_lender['COMPANY NAME'])

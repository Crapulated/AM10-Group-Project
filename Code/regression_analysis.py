import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 1. LOAD AND CLEAN DATA
# ====================================================================
print("="*70)
print("LOADING AND CLEANING DATA")
print("="*70)

df = pd.read_csv("../data/movie_metadata.csv")
print(f"\nOriginal shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# Target variable
target_column = "imdb_score"

# Drop rows with missing target
df = df.dropna(subset=[target_column])
print(f"After dropping missing target: {df.shape}")

# Select only numeric features
numeric_df = df.select_dtypes(include=['float64', 'int64'])
X = numeric_df.drop(columns=[target_column])
y = df[target_column]

# Remove columns with too many missing values (>30%)
missing_pct = X.isnull().sum() / len(X)
cols_to_keep = missing_pct[missing_pct < 0.3].index
X = X[cols_to_keep]
print(f"Columns after removing high-missing features: {X.shape[1]}")

# Fill remaining missing values with median (more robust than mean)
X = X.fillna(X.median())

# Remove constant or near-constant columns
nunique = X.nunique()
X = X.loc[:, nunique > 1]
print(f"Columns after removing constants: {X.shape[1]}")

# Remove highly correlated features (correlation > 0.95)
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [column for column in upper_triangle.columns 
           if any(upper_triangle[column] > 0.95)]
X = X.drop(columns=to_drop)
print(f"Columns after removing highly correlated features: {X.shape[1]}")
print(f"\nFinal feature set: {list(X.columns)}\n")

# ====================================================================
# 2. TRAIN-TEST SPLIT
# ====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for statsmodels
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# ====================================================================
# 3. STEPWISE REGRESSION (Backward Elimination)
# ====================================================================
print("="*70)
print("STEPWISE REGRESSION (Backward Elimination)")
print("="*70)

def backward_elimination(X, y, significance_level=0.05):
    """
    Perform backward elimination to select features
    """
    X_with_const = sm.add_constant(X)
    features = list(X.columns)
    
    while len(features) > 0:
        # Fit model
        model = sm.OLS(y, X_with_const[['const'] + features]).fit()
        
        # Get p-values (excluding constant)
        p_values = model.pvalues[1:]
        max_p_value = p_values.max()
        
        if max_p_value > significance_level:
            # Remove feature with highest p-value
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
            print(f"Removed: {excluded_feature} (p-value: {max_p_value:.4f})")
        else:
            break
    
    return features

print("\nPerforming backward elimination (p-value threshold: 0.05)...\n")
selected_features = backward_elimination(X_train_scaled_df, y_train)

print(f"\n{len(selected_features)} features selected: {selected_features}\n")

# Fit final OLS model with selected features
X_train_selected = sm.add_constant(X_train_scaled_df[selected_features])
X_test_selected = sm.add_constant(X_test_scaled_df[selected_features])

ols_model = sm.OLS(y_train, X_train_selected).fit()

# ====================================================================
# 4. OLS RESULTS
# ====================================================================
print("="*70)
print("OLS REGRESSION RESULTS (After Stepwise Selection)")
print("="*70)
print(ols_model.summary())

# Predictions
y_train_pred_ols = ols_model.predict(X_train_selected)
y_test_pred_ols = ols_model.predict(X_test_selected)

train_r2_ols = r2_score(y_train, y_train_pred_ols)
test_r2_ols = r2_score(y_test, y_test_pred_ols)
train_mse_ols = mean_squared_error(y_train, y_train_pred_ols)
test_mse_ols = mean_squared_error(y_test, y_test_pred_ols)
test_rmse_ols = np.sqrt(test_mse_ols)

# Check for multicollinearity (VIF)
print("\n" + "="*70)
print("VARIANCE INFLATION FACTORS (VIF)")
print("="*70)
vif_data = pd.DataFrame()
vif_data["Feature"] = selected_features
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled_df[selected_features].values, i) 
                   for i in range(len(selected_features))]
print(vif_data.to_string(index=False))
print("\nNote: VIF > 10 indicates high multicollinearity\n")

# ====================================================================
# 5. LASSO REGRESSION (L1 Regularization)
# ====================================================================
print("="*70)
print("LASSO REGRESSION (L1 Regularization)")
print("="*70)

# Use cross-validation to find optimal alpha
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha (via CV): {lasso_cv.alpha_:.6f}")

# Fit final Lasso model
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_lasso = lasso.predict(X_train_scaled)
y_test_pred_lasso = lasso.predict(X_test_scaled)

train_r2_lasso = r2_score(y_train, y_train_pred_lasso)
test_r2_lasso = r2_score(y_test, y_test_pred_lasso)
train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
test_rmse_lasso = np.sqrt(test_mse_lasso)

# Feature selection by Lasso
lasso_coefs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
})
lasso_coefs = lasso_coefs[lasso_coefs['Coefficient'] != 0].sort_values(
    by='Coefficient', key=abs, ascending=False
)

print(f"\nNumber of non-zero coefficients: {len(lasso_coefs)} out of {len(X.columns)}")
print("\nTop features selected by Lasso:")
print(lasso_coefs.to_string(index=False))

# ====================================================================
# 6. RIDGE REGRESSION (L2 Regularization)
# ====================================================================
print("\n" + "="*70)
print("RIDGE REGRESSION (L2 Regularization)")
print("="*70)

# Use cross-validation to find optimal alpha
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha (via CV): {ridge_cv.alpha_:.6f}")

# Fit final Ridge model
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_ridge = ridge.predict(X_train_scaled)
y_test_pred_ridge = ridge.predict(X_test_scaled)

train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)
train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
test_rmse_ridge = np.sqrt(test_mse_ridge)

# Show top features by absolute coefficient value
ridge_coefs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': ridge.coef_
})
ridge_coefs = ridge_coefs.sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop features by Ridge coefficient magnitude:")
print(ridge_coefs.head(15).to_string(index=False))

# ====================================================================
# 7. FINAL SUMMARY
# ====================================================================
print("\n" + "="*70)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*70)

summary_df = pd.DataFrame({
    'Model': ['OLS (Stepwise)', 'Lasso', 'Ridge'],
    'Features Used': [len(selected_features), len(lasso_coefs), len(X.columns)],
    'Train R²': [train_r2_ols, train_r2_lasso, train_r2_ridge],
    'Test R²': [test_r2_ols, test_r2_lasso, test_r2_ridge],
    'Train MSE': [train_mse_ols, train_mse_lasso, train_mse_ridge],
    'Test MSE': [test_mse_ols, test_mse_lasso, test_mse_ridge],
    'Test RMSE': [test_rmse_ols, test_rmse_lasso, test_rmse_ridge],
})

print("\n" + summary_df.to_string(index=False))

# Determine best model
best_model_idx = summary_df['Test R²'].idxmax()
best_model_name = summary_df.loc[best_model_idx, 'Model']

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name} (highest Test R²)")
print(f"{'='*70}")

# Additional insights
print("\nKEY INSIGHTS:")
print(f"1. OLS with stepwise selection reduced features from {len(X.columns)} to {len(selected_features)}")
print(f"2. Lasso selected {len(lasso_coefs)} features (most sparse model)")
print(f"3. Ridge uses all {len(X.columns)} features but shrinks coefficients")

overfit_check_ols = train_r2_ols - test_r2_ols
overfit_check_lasso = train_r2_lasso - test_r2_lasso
overfit_check_ridge = train_r2_ridge - test_r2_ridge

print(f"\nOverfitting Check (Train R² - Test R²):")
print(f"  OLS:   {overfit_check_ols:.4f}")
print(f"  Lasso: {overfit_check_lasso:.4f}")
print(f"  Ridge: {overfit_check_ridge:.4f}")
print(f"  (Lower is better - indicates less overfitting)\n")

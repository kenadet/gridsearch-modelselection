import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("housing")

# 2. Create dependent and independent variables
X = df.drop(columns= 'price') # Create predictors or independent variables or Otherwise called fetures
y = df['price'] # Create target or dependent variable

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Proprocessing pipelines
num_features = ['age', 'size'] # Or numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
cat_features = ['region'] # categorical_features = df.select_dtypes(include=['object', 'category']).columns


num_pipeline = Pipeline([   # If there are outliers detected during describe, and EDA, we would need to remove them first before doing this
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy = 'most_frequent')),
  ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
  ('num', num_pipeline, num_features), 
  ('cat',cat_pipeline, cat_features )
])

# 5. Ridge Regression anchoring the processor
ridge_pipeline = Pipeline([
  ('processing', preprocessor),
  ('regressor', Ridge())
])

# 6. Gridsearch setup to find best parameter of Ridge
param_grid = {'regressor__alpha': [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')  
# Cross validate rotate the train data around subsets to find best model with best perform (Normal fitting),
# Grisearch find best hyperparameter that does this.
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['regressor__alpha']
best_cv_mse = -grid_search.best_score_

print(f"Ridge Gridsearch and Cross Validation Best alpha: {best_alpha}")
print(f"Ridge Gridsearch and Cross Validation Best cross-validated MSE: {best_cv_mse:.2f}")

# 7. Use best Ridge model on training  set
final_ridge = Pipeline ([
    ('preprocessing', preprocessor),
    ('regressor', Ridge(alpha=best_alpha))
])
final_ridge.fit(X_train, y_train)


# 8. Predict and Evaluate on test set
y_pred_ridge = final_ridge.predict(X_test)
test_mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Ridge Test MSE: {test_mse_ridge:.2f}")

# 9 Compare with Dummy Regressor and RandomForest
dummy = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', DummyRegressor(strategy='mean'))
])
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
test_mse_dummy = mean_squared_error(y_test, y_pred_dummy)
print(f"Baseline Test MSE: {test_mse_dummy:.2f}")


rf = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
test_mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Test MSE: {test_mse_rf:.2f}")






import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler,Normalizer,RobustScaler

# load data
df = pd.read_csv('train.csv')
tdf = pd.read_csv('test.csv')

# check for missing values
print(df.isnull().sum())

# remove duplicates
df.drop_duplicates()

# split data into features and target
X = df.drop(['id','yield'],axis=1)
y = df['yield'].values

# split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
Xt = scaler.transform(tdf.drop(['id'],axis=1))

# build pipeline with PCA and Gradient Boosting Regressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
pipe = Pipeline([
    ('pca', PCA()),
    ('gbr', GradientBoostingRegressor(
        loss='squared_error',
        random_state=42
    ))
])

# set up grid search parameters
params = {
    'pca__n_components': [5, 10, 15],
    'gbr__n_estimators': [100, 500, 1000],
    'gbr__learning_rate': [0.1, 0.01, 0.001],
    'gbr__max_depth': [3, 5, 7],
    'gbr__subsample': [0.5, 0.75, 1.0],
    'gbr__alpha': [0.01, 0.1, 1]
}

# perform grid search
grid = GridSearchCV(pipe, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# make predictions on validation set
y_val_pred = grid.predict(X_val)

# evaluate performance on validation set
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
print(f'Validation set RMSE: {rmse}')

# make predictions on test set
y_pred = grid.predict(Xt)

# write predictions to file
ndf=pd.DataFrame({"id":tdf['id'],"yield":y_pred})
ndf.to_csv('result.csv',index=False)

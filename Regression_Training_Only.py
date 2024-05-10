from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

import time
# Record the start time
start_time = time.time()

data_frame = pd.read_csv('Regression_first_7000_rows.csv')


# data_frame['time'] = pd.to_datetime(data_frame['time'], unit='s')
# data_frame['time'] = pd.to_datetime(data_frame['time'], unit='s')
data_frame.drop('price',axis=1, inplace=True)
data_frame['currency'] = 0
data_frame['fee'] = 0
columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source')

data_frame['amenities'] = data_frame['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = data_frame['amenities'].str.get_dummies(sep=',')
amenities_columns=df_encoded_amenities.columns.values

data_frame.drop(columns=['amenities'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_amenities], axis=1)

null_replacement={}
null_replacement['address']=data_frame['address'].mode()[0]
null_replacement['cityname']=data_frame['cityname'].mode()[0]
null_replacement['state']=data_frame['state'].mode()[0]
null_replacement['bathrooms']=data_frame['bathrooms'].mean()
null_replacement['bedrooms']=data_frame['bedrooms'].mean()
null_replacement['latitude']=data_frame['latitude'].mean()
null_replacement['longitude']=data_frame['longitude'].mean()

for col in null_replacement:
    data_frame[col] = data_frame[col].fillna(null_replacement[col])

data_frame['price_display'] = data_frame['price_display'].str.replace(r'[^\d]', '', regex=True).astype(int)


df_encoded_pets_allowed = data_frame['pets_allowed'].str.get_dummies(sep=',')
pet_columns=df_encoded_pets_allowed.columns.values

data_frame.drop(columns=['pets_allowed'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_pets_allowed], axis=1)


state_mean_price = data_frame.groupby('state')['price_display'].mean()
data_frame['state_mean_price'] = data_frame['state'].map(state_mean_price)
city_mean_price = data_frame.groupby('cityname')['price_display'].mean()
data_frame['city_mean_price'] = data_frame['cityname'].map(city_mean_price)
data_frame.drop(['state', 'cityname'], axis=1, inplace=True)



cats_mean_price = data_frame.groupby('Cats')['price_display'].mean()
data_frame['cats_mean_price'] = data_frame['Cats'].map(cats_mean_price)
dogs_mean_price = data_frame.groupby('Dogs')['price_display'].mean()
data_frame['dogs_mean_price'] = data_frame['Dogs'].map(dogs_mean_price)
data_frame.drop(['Cats', 'Dogs'], axis=1, inplace=True)

data_frame['sum4'] = data_frame['state_mean_price']  + data_frame['city_mean_price']  + (data_frame['cats_mean_price']  + data_frame['dogs_mean_price'] )
data_frame['sum5'] = data_frame['state_mean_price']  * data_frame['city_mean_price'] + (data_frame['cats_mean_price']  * data_frame['dogs_mean_price'] )

label_encoders = {} #Dictionary to store label encoders
for c in columns_for_encoding:
    lbl = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    lbl.fit(data_frame[[c]])
    data_frame[c] = lbl.transform(data_frame[[c]])
    label_encoders[c] = lbl


from scipy import stats
z_threshold = 3
with np.errstate(divide='ignore', invalid='ignore'):
    z_scores = stats.zscore(data_frame)
z_scores = np.nan_to_num(z_scores)
outliers_mask = (z_scores < -z_threshold) | (z_scores > z_threshold)
data_frame_no_outliers = data_frame[~outliers_mask.any(axis=1)]
data_frame = data_frame_no_outliers





scaled_cols = list(data_frame.columns)
scaled_cols.remove("price_display")
scaler = StandardScaler()
data_frame[scaled_cols] = scaler.fit_transform(data_frame[scaled_cols])
data_frame.head()


plt.scatter(data_frame['city_mean_price'], data_frame['price_display'])
plt.xlabel('city_mean_price')
plt.ylabel('price_display')
plt.title('Scatter plot of city_mean_price vs price_display')
plt.show()
plt.scatter(data_frame['square_feet'], data_frame['price_display'])
plt.xlabel('square_feet')
plt.ylabel('price_display')
plt.title('Scatter plot of square_feet vs price_display')
plt.show()
plt.scatter(data_frame['bedrooms'], data_frame['price_display'])
plt.xlabel('bedrooms')
plt.ylabel('price_display')
plt.title('Scatter plot of bedrooms vs price_display')
plt.show()


plt.subplots(figsize=(12, 8))
corr = data_frame.corr()
top_feature = list(corr.index[abs(corr['price_display'])>0.2])
top_corr = data_frame[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature.remove('price_display')
X = data_frame.drop('price_display', axis=1)
X=data_frame[top_feature]
y = data_frame['price_display']

def evaluate_model(model, X_train, y_train, X_test, y_test):
  # Fit the model to the training data
  model.fit(X_train, y_train)
  # Make predictions on both training and testing sets
  train_predictions = model.predict(X_train)
  test_predictions = model.predict(X_test)
  # Calculate Mean Squared Error for both training and testing sets
  from sklearn.metrics import mean_squared_error
  train_mse = mean_squared_error(y_train, train_predictions)
  test_mse = mean_squared_error(y_test, test_predictions)
  # Calculate R-squared score for both training and testing sets
  from sklearn.metrics import r2_score
  train_r2 = r2_score(y_train, train_predictions)
  test_r2 = r2_score(y_test, test_predictions)
  # Print evaluation metrics
  print('Model Train MSE:', train_mse)
  print('Model Test MSE:', test_mse)
  print('Model Train R-squared score:', train_r2)
  print('Model Test R-squared score:', test_r2)




poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
poly_model = linear_model.LinearRegression()
print("Polynomial model")
evaluate_model(poly_model,X,y,X,y)
print("______________________")


lr = linear_model.LinearRegression()
print("Linear regression model")
evaluate_model(lr,X,y,X,y)
print("______________________")



from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
print("Random Forest model")
evaluate_model(rf_regressor,X,y,X,y)
print("______________________")




from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)  # Specify the alpha values to try
print("Lasso CV model")
evaluate_model(lasso_cv,X,y,X,y)
print("______________________")






from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0])
print("Ridge CV model")
evaluate_model(ridge_cv,X,y,X,y)
print("______________________")







from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=7)
print("Decision tree model")
evaluate_model(model,X,y,X,y)
print("______________________")




from sklearn.ensemble import GradientBoostingRegressor

gb_regressor = GradientBoostingRegressor()
print("Gradient boosting regressor model")
evaluate_model(gb_regressor,X,y,X,y)
print("______________________")


# Record the end time
end_time = time.time()

# Calculate the difference
time_difference = end_time - start_time

print("Time taken:", time_difference, "seconds")

import pickle
with open("regression.pkl", "wb") as f:
        pickle.dump(null_replacement, f)
        pickle.dump(amenities_columns, f)
        pickle.dump(pet_columns, f)
        pickle.dump(state_mean_price, f)
        pickle.dump(city_mean_price, f)
        pickle.dump(cats_mean_price, f)
        pickle.dump(dogs_mean_price, f)
        pickle.dump(label_encoders, f)
        pickle.dump(scaled_cols, f)
        pickle.dump(scaler, f)
        pickle.dump(top_feature, f)
        pickle.dump(gb_regressor, f)


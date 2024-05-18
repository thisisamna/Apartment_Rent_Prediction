from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split





data_frame = pd.read_csv('Regression_Dataset.csv')
y=data_frame['price_display']
X=data_frame.drop(columns=['price_display'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,shuffle=True,random_state=10)
train_data_frame=pd.DataFrame(X_train)
train_data_frame['price_display']=y_train



train_data_frame.drop('price',axis=1, inplace=True)
train_data_frame['currency'] = 0
train_data_frame['fee'] = 0
columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source')

train_data_frame['amenities'] = train_data_frame['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = train_data_frame['amenities'].str.get_dummies(sep=',')
amenities_columns=df_encoded_amenities.columns.values

train_data_frame.drop(columns=['amenities'], inplace=True)
train_data_frame = pd.concat([train_data_frame, df_encoded_amenities], axis=1)

null_replacement={}
null_replacement['address']=train_data_frame['address'].mode()[0]
null_replacement['cityname']=train_data_frame['cityname'].mode()[0]
null_replacement['state']=train_data_frame['state'].mode()[0]
null_replacement['bathrooms']=train_data_frame['bathrooms'].mean()
null_replacement['bedrooms']=train_data_frame['bedrooms'].mean()
null_replacement['latitude']=train_data_frame['latitude'].mean()
null_replacement['longitude']=train_data_frame['longitude'].mean()

for col in null_replacement:
    train_data_frame[col] = train_data_frame[col].fillna(null_replacement[col])

train_data_frame['price_display'] = train_data_frame['price_display'].str.replace(r'[^\d]', '', regex=True).astype(int)


df_encoded_pets_allowed = train_data_frame['pets_allowed'].str.get_dummies(sep=',')
pet_columns=df_encoded_pets_allowed.columns.values

train_data_frame.drop(columns=['pets_allowed'], inplace=True)
train_data_frame = pd.concat([train_data_frame, df_encoded_pets_allowed], axis=1)


state_mean_price = train_data_frame.groupby('state')['price_display'].mean()
train_data_frame['state_mean_price'] = train_data_frame['state'].map(state_mean_price)
city_mean_price = train_data_frame.groupby('cityname')['price_display'].mean()
train_data_frame['city_mean_price'] = train_data_frame['cityname'].map(city_mean_price)
train_data_frame.drop(['state', 'cityname'], axis=1, inplace=True)



cats_mean_price = train_data_frame.groupby('Cats')['price_display'].mean()
train_data_frame['cats_mean_price'] = train_data_frame['Cats'].map(cats_mean_price)
dogs_mean_price = train_data_frame.groupby('Dogs')['price_display'].mean()
train_data_frame['dogs_mean_price'] = train_data_frame['Dogs'].map(dogs_mean_price)
train_data_frame.drop(['Cats', 'Dogs'], axis=1, inplace=True)

train_data_frame['sum4'] = train_data_frame['state_mean_price']  + train_data_frame['city_mean_price']  + (train_data_frame['cats_mean_price']  + train_data_frame['dogs_mean_price'] )
train_data_frame['sum5'] = train_data_frame['state_mean_price']  * train_data_frame['city_mean_price'] + (train_data_frame['cats_mean_price']  * train_data_frame['dogs_mean_price'] )
train_data_frame['ratio_bedrooms_bathrooms'] = train_data_frame['bedrooms'] / train_data_frame['bathrooms']

label_encoders = {} #Dictionary to store label encoders
for c in columns_for_encoding:
    lbl = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    lbl.fit(train_data_frame[[c]])
    train_data_frame[c] = lbl.transform(train_data_frame[[c]])
    label_encoders[c] = lbl


# from scipy import stats
# z_threshold = 3
# with np.errstate(divide='ignore', invalid='ignore'):
#     z_scores = stats.zscore(train_data_frame)
# z_scores = np.nan_to_num(z_scores)
# outliers_mask = (z_scores < -z_threshold) | (z_scores > z_threshold)
# train_data_frame_no_outliers = train_data_frame[~outliers_mask.any(axis=1)]
# train_data_frame = train_data_frame_no_outliers





scaled_cols = list(train_data_frame.columns)
scaled_cols.remove("price_display")
scaler = MinMaxScaler()
train_data_frame[scaled_cols] = scaler.fit_transform(train_data_frame[scaled_cols])
train_data_frame.head()


plt.scatter(train_data_frame['city_mean_price'], train_data_frame['price_display'])
plt.xlabel('city_mean_price')
plt.ylabel('price_display')
plt.title('Scatter plot of city_mean_price vs price_display')
plt.show()
plt.scatter(train_data_frame['square_feet'], train_data_frame['price_display'])
plt.xlabel('square_feet')
plt.ylabel('price_display')
plt.title('Scatter plot of square_feet vs price_display')
plt.show()
plt.scatter(train_data_frame['bedrooms'], train_data_frame['price_display'])
plt.xlabel('bedrooms')
plt.ylabel('price_display')
plt.title('Scatter plot of bedrooms vs price_display')
plt.show()



X_train = train_data_frame.drop('price_display', axis=1)
y_train = train_data_frame['price_display']

from sklearn.feature_selection import mutual_info_regression

# Calculate Information Gain for each feature
info_gain = mutual_info_regression(X_train, y_train)

# Select the top features based on Information Gain
num_features_to_select = 30  # Change this number as per your preference
selected_features_indices = (-info_gain).argsort()[:num_features_to_select]
selected_features = X_train.columns[selected_features_indices]

print(selected_features)

X_train = X_train[selected_features]

#-------------------------------------------------------
#Preprocessing test data

X_test['currency'] = 0
X_test['fee'] = 0

for col in null_replacement:
    X_test[col] = X_test[col].fillna(null_replacement[col])

X_test['amenities'] = X_test['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = X_test['amenities'].str.get_dummies(sep=',')
amenities_columns=df_encoded_amenities.columns.values
X_test.drop(columns=['amenities'], inplace=True)
X_test = pd.concat([X_test, df_encoded_amenities], axis=1)
#Check if a column is not found in test data 
for col in amenities_columns:
    if col not in X_test:
        X_test[col]=np.zeroes_like()

df_encoded_pets_allowed = X_test['pets_allowed'].str.get_dummies(sep=',')
X_test.drop(columns=['pets_allowed'], inplace=True)
X_test = pd.concat([X_test, df_encoded_pets_allowed], axis=1)
#Check if a column is not found in test data 
for col in pet_columns:
    if col not in X_test:
        X_test[col]=np.zeroes_like()



y_test = y_test.str.replace(r'[^\d]', '', regex=True).astype(int)

X_test['state_mean_price'] = X_test['state'].map(state_mean_price)
X_test['city_mean_price'] = X_test['cityname'].map(city_mean_price)
X_test.drop(['state', 'cityname'], axis=1, inplace=True)
#Fill nulls caused by unseen state
X_test['state_mean_price'] = X_test['state_mean_price'].fillna(0)
X_test['city_mean_price'] = X_test['city_mean_price'].fillna(0)

X_test['cats_mean_price'] = X_test['Cats'].map(cats_mean_price)
X_test['dogs_mean_price'] = X_test['Dogs'].map(dogs_mean_price)
X_test.drop(['Cats', 'Dogs'], axis=1, inplace=True)

columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source')
for c in columns_for_encoding:
    X_test[c] = label_encoders[c].transform(X_test[[c]])

X_test['sum4'] = X_test['state_mean_price']  + X_test['city_mean_price']  + (X_test['cats_mean_price']  + X_test['dogs_mean_price'] )
X_test['sum5'] = X_test['state_mean_price']  * X_test['city_mean_price'] + (X_test['cats_mean_price']  * X_test['dogs_mean_price'] )
X_test['ratio_bedrooms_bathrooms'] = X_test['bedrooms'] / X_test['bathrooms']

X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])

X_test = X_test[selected_features]


#-----------------------------------------------------

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
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
print("Polynomial model")
evaluate_model(poly_model,X_train,y_train,X_test,y_test)
print("______________________")


lr = linear_model.LinearRegression()
print("Linear regression model")
evaluate_model(lr,X_train,y_train,X_test,y_test)
print("______________________")



from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
print("Random Forest model")
evaluate_model(rf_regressor,X_train,y_train,X_test,y_test)
print("______________________")




from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)  # Specify the alpha values to try
print("Lasso CV model")
evaluate_model(lasso_cv,X_train,y_train,X_test,y_test)
print("______________________")






from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0])
print("Ridge CV model")
evaluate_model(ridge_cv,X_train,y_train,X_test,y_test)
print("______________________")







from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=7)
print("Decision tree model")
evaluate_model(model,X_train,y_train,X_test,y_test)
print("______________________")




from sklearn.ensemble import GradientBoostingRegressor

gb_regressor = GradientBoostingRegressor(learning_rate=0.2,max_depth=3)
print("Gradient boosting regressor model")
evaluate_model(gb_regressor,X_train,y_train,X_test,y_test)
print("______________________")






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
        pickle.dump(selected_features, f)
        pickle.dump(poly_model, f)
        pickle.dump(lr, f)
        pickle.dump(rf_regressor, f)
        pickle.dump(lasso_cv, f)
        pickle.dump(ridge_cv, f)
        pickle.dump(model, f)
        pickle.dump(gb_regressor, f)


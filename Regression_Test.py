from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

with open("regression.pkl", "rb") as f:
        null_replacement=pickle.load(f)
        amenities_columns=pickle.load(f)
        pet_columns=pickle.load(f)
        state_mean_price=pickle.load(f)
        city_mean_price=pickle.load(f)
        cats_mean_price=pickle.load(f)
        dogs_mean_price=pickle.load(f)
        label_encoders=pickle.load(f)
        scaled_cols=pickle.load(f)
        scaler=pickle.load(f)
        top_feature=pickle.load(f)
        poly_model=pickle.load(f)
        lr=pickle.load(f)
        rf_regressor=pickle.load(f)
        lasso_cv=pickle.load(f)
        ridge_cv=pickle.load(f)
        model=pickle.load(f)
        gb_regressor=pickle.load(f)


data_frame = pd.read_csv('ApartmentRentPrediction.csv')

data_frame.drop('price',axis=1, inplace=True)
data_frame['currency'] = 0
data_frame['fee'] = 0

for col in null_replacement:
    data_frame[col] = data_frame[col].fillna(null_replacement[col])

data_frame['amenities'] = data_frame['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = data_frame['amenities'].str.get_dummies(sep=',')
amenities_columns=df_encoded_amenities.columns.values
data_frame.drop(columns=['amenities'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_amenities], axis=1)
#Check if a column is not found in test data 
for col in amenities_columns:
    if col not in data_frame:
        data_frame[col]=np.zeroes_like()

df_encoded_pets_allowed = data_frame['pets_allowed'].str.get_dummies(sep=',')
data_frame.drop(columns=['pets_allowed'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_pets_allowed], axis=1)
#Check if a column is not found in test data 
for col in pet_columns:
    if col not in data_frame:
        data_frame[col]=np.zeroes_like()



data_frame['price_display'] = data_frame['price_display'].str.replace(r'[^\d]', '', regex=True).astype(int)

data_frame['state_mean_price'] = data_frame['state'].map(state_mean_price)
data_frame['city_mean_price'] = data_frame['cityname'].map(city_mean_price)
data_frame.drop(['state', 'cityname'], axis=1, inplace=True)
#Fill nulls caused by unseen state
data_frame['state_mean_price'] = data_frame['state_mean_price'].fillna(0)
data_frame['city_mean_price'] = data_frame['city_mean_price'].fillna(0)

data_frame['cats_mean_price'] = data_frame['Cats'].map(cats_mean_price)
data_frame['dogs_mean_price'] = data_frame['Dogs'].map(dogs_mean_price)
data_frame.drop(['Cats', 'Dogs'], axis=1, inplace=True)

columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source')
for c in columns_for_encoding:
    data_frame[c] = label_encoders[c].transform(data_frame[[c]])

data_frame['sum4'] = data_frame['state_mean_price']  + data_frame['city_mean_price']  + (data_frame['cats_mean_price']  + data_frame['dogs_mean_price'] )
data_frame['sum5'] = data_frame['state_mean_price']  * data_frame['city_mean_price'] + (data_frame['cats_mean_price']  * data_frame['dogs_mean_price'] )


data_frame[scaled_cols] = scaler.transform(data_frame[scaled_cols])

X = data_frame[top_feature]
y = data_frame['price_display']


def evaluate_model(model, X, y):
  # Fit the model to the training data  # Make predictions on both training and testing sets
  predictions = model.predict(X)
  # Calculate Mean Squared Error for both training and testing sets
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y, predictions)
  # Calculate R-squared score for both training and testing sets
  from sklearn.metrics import r2_score
  r2 = r2_score(y, predictions)
  # Print evaluation metrics
  print('Model MSE:', mse)
  print('Model R-squared score:', r2)


print("Polynomial model")
evaluate_model(poly_model,X,y)
print("______________________")


print("Linear regression model")
evaluate_model(lr,X,y)
print("______________________")


print("Random Forest model")
evaluate_model(rf_regressor,X,y)
print("______________________")


print("Lasso CV model")
evaluate_model(lasso_cv,X,y)
print("______________________")



print("Ridge CV model")
evaluate_model(ridge_cv,X,y)
print("______________________")


print("Decision tree model")
evaluate_model(model,X,y)
print("______________________")


print("Gradient boosting regressor model")
evaluate_model(gb_regressor,X,y)
print("______________________")


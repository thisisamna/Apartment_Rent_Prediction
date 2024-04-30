from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

data_frame = pd.read_csv('ApartmentRentPrediction.csv')

data_frame.drop('price',axis=1, inplace=True)
data_frame['currency'] = 0
data_frame['fee'] = 0

null_replacement={}
for col in null_replacement:
    data_frame[col] = data_frame[col].fillna(null_replacement[col])

data_frame['amenities'] = data_frame['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = data_frame['amenities'].str.get_dummies(sep=',')
amenities_columns=df_encoded_amenities.columns.values
data_frame.drop(columns=['amenities'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_amenities], axis=1)
amenities_columns=???
#Check if a column is not found in test data 
for col in amenities_columns:
    if col not in data_frame:
        data_frame[col]=np.zeroes_like()

df_encoded_pets_allowed = data_frame['pets_allowed'].str.get_dummies(sep=',')
data_frame.drop(columns=['pets_allowed'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_pets_allowed], axis=1)
pet_columns=???
#Check if a column is not found in test data 
for col in pet_columns:
    if col not in data_frame:
        data_frame[col]=np.zeroes_like()



data_frame['price_display'] = data_frame['price_display'].str.replace(r'[^\d]', '', regex=True).astype(int)

state_mean_price=???
data_frame['state_mean_price'] = data_frame['state'].map(state_mean_price)
ciry_mean_price=???
data_frame['city_mean_price'] = data_frame['cityname'].map(city_mean_price)
data_frame.drop(['state', 'cityname'], axis=1, inplace=True)

cats_mean_price=???
data_frame['cats_mean_price'] = data_frame['Cats'].map(cats_mean_price)
dogs_mean_price=???
data_frame['dogs_mean_price'] = data_frame['Dogs'].map(dogs_mean_price)
data_frame.drop(['Cats', 'Dogs'], axis=1, inplace=True)

columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source')
label_encoders = {} ???
for c in columns_for_encoding:
    data_frame[c] = label_encoders[c].transform(data_frame[[c]])

data_frame['sum4'] = data_frame['state_mean_price']  + data_frame['city_mean_price']  + (data_frame['cats_mean_price']  + data_frame['dogs_mean_price'] )
data_frame['sum5'] = data_frame['state_mean_price']  * data_frame['city_mean_price'] + (data_frame['cats_mean_price']  * data_frame['dogs_mean_price'] )


scaled_cols=???
scaler=???
data_frame[scaled_cols] = scaler.transform(data_frame[scaled_cols])


X = data_frame[top_feature]
X = X.drop('price_display', axis=1)
Y = data_frame['price_display']

gb_regressor=???
evaluate_model(gb_regressor,X_train,y_train,X_test,y_test)

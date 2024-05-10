from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split


data_frame = pd.read_csv('ApartmentRentPrediction_Milestone2.csv')


# data_frame['time'] = pd.to_datetime(data_frame['time'], unit='s')
# data_frame['time'] = pd.to_datetime(data_frame['time'], unit='s')
data_frame['currency'] = 0
data_frame['fee'] = 0
columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source', 'state_mode_price','city_mode_price', 'cats_mode_price','dogs_mode_price')

data_frame['amenities'] = data_frame['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = data_frame['amenities'].str.get_dummies(sep=',')
amenities_columns=df_encoded_amenities.columns.values

data_frame.drop(columns=['amenities'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_amenities], axis=1)
print(data_frame.info())
null_replacement={}
null_replacement['address']=data_frame['address'].mode()[0]
null_replacement['cityname']=data_frame['cityname'].mode()[0]
null_replacement['state']=data_frame['state'].mode()[0]
null_replacement['bathrooms']=data_frame['bathrooms'].mean()
null_replacement['bedrooms']=data_frame['bedrooms'].mean()
null_replacement['latitude']=data_frame['latitude'].mean()
null_replacement['longitude']=data_frame['longitude'].mean()
null_replacement['pets_allowed']=""

for col in null_replacement:
    data_frame[col] = data_frame[col].fillna(null_replacement[col])
print(data_frame.info())
df_encoded_pets_allowed = data_frame['pets_allowed'].str.get_dummies(sep=',')
pet_columns=df_encoded_pets_allowed.columns.values

data_frame.drop(columns=['pets_allowed'], inplace=True)
data_frame = pd.concat([data_frame, df_encoded_pets_allowed], axis=1)


state_mode_price = data_frame.groupby('state')['RentCategory'].agg(lambda x: x.value_counts().index[0])
data_frame['state_mode_price'] = data_frame['state'].map(state_mode_price)
city_mode_price = data_frame.groupby('cityname')['RentCategory'].agg(lambda x: x.value_counts().index[0])
data_frame['city_mode_price'] = data_frame['cityname'].map(city_mode_price)
data_frame.drop(['state', 'cityname'], axis=1, inplace=True)



cats_mode_price = data_frame.groupby('Cats')['RentCategory'].agg(lambda x: x.value_counts().index[0])
data_frame['cats_mode_price'] = data_frame['Cats'].map(cats_mode_price)
dogs_mode_price = data_frame.groupby('Dogs')['RentCategory'].agg(lambda x: x.value_counts().index[0])
data_frame['dogs_mode_price'] = data_frame['Dogs'].map(dogs_mode_price)
data_frame.drop(['Cats', 'Dogs'], axis=1, inplace=True)

label_encoders = {} #Dictionary to store label encoders
for c in columns_for_encoding:
    lbl = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    lbl.fit(data_frame[[c]])
    data_frame[c] = lbl.transform(data_frame[[c]])
    label_encoders[c] = lbl







scaled_cols = list(data_frame.columns)
scaled_cols.remove("RentCategory")
scaler = StandardScaler()
data_frame[scaled_cols] = scaler.fit_transform(data_frame[scaled_cols])
data_frame.head()


plt.scatter(data_frame['city_mode_price'], data_frame['RentCategory'])
plt.xlabel('city_mode_price')
plt.ylabel('RentCategory')
plt.title('Scatter plot of city_mode_price vs RentCategory')
plt.show()
plt.scatter(data_frame['square_feet'], data_frame['RentCategory'])
plt.xlabel('square_feet')
plt.ylabel('RentCategory')
plt.title('Scatter plot of square_feet vs RentCategory')
plt.show()
plt.scatter(data_frame['bedrooms'], data_frame['RentCategory'])
plt.xlabel('bedrooms')
plt.ylabel('RentCategory')
plt.title('Scatter plot of bedrooms vs RentCategory')
plt.show()


# plt.subplots(figsize=(12, 8))
# corr = data_frame.corr()
# top_feature = corr.index[abs(corr['RentCategory'])>0.2]
# top_corr = data_frame[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()

#X = data_frame[top_feature]


X = data_frame.drop('RentCategory', axis=1)
Y = data_frame['RentCategory']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
print(X.info())

#Random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) #ttghiar
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred_RFtrain = rf_classifier.predict(X_train)
y_pred_RFtest = rf_classifier.predict(X_test)

# Evaluation
accuracy_train = rf_classifier.score(X_train, y_train)
accuracy_test = rf_classifier.score(X_test, y_test)
print('\nRandom Forest train accuracy:', accuracy_train)
print('Random Forest test accuracy:', accuracy_test)
print('\nClassification Report of Random Forest:')
print(metrics.classification_report(y_test, y_pred_RFtest))
print('\nConfusion Matrix of Random Forest:')
print(metrics.confusion_matrix(y_test, y_pred_RFtest))

#------------------------------------------------------------------
#SVM
from sklearn.svm import SVC
from sklearn import metrics

svm_classifier = SVC(kernel='rbf', random_state=42) #ttghiar
svm_classifier.fit(X_train, y_train)

# Predictions
y_pred_svmtrain = svm_classifier.predict(X_train)
y_pred_svmtest = svm_classifier.predict(X_test)

# Evaluation
accuracy_train = svm_classifier.score(X_train, y_train)
accuracy_test = svm_classifier.score(X_test, y_test)
print('\n SVM train accuracy:', accuracy_train)
print('SVM  test accuracy:', accuracy_test)
print('\nClassification Report of SVM:')
print(metrics.classification_report(y_test, y_pred_svmtest))
print('\nConfusion Matrix of SVM:')
print(metrics.confusion_matrix(y_test, y_pred_svmtest))


#------------------------------------------------------------------
#LOGISTIC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logistic_classifier = LogisticRegression(random_state=42) #ttghiar
logistic_classifier.fit(X_train, y_train)

# Predictions
y_pred_logistictrain = logistic_classifier.predict(X_train)
y_pred_logistictest = logistic_classifier.predict(X_test)

# Evaluation
accuracy_train = logistic_classifier.score(X_train, y_train)
accuracy_test = logistic_classifier.score(X_test, y_test)
print('\nLogistic Regression train accuarcy:', accuracy_train)
print('Logistic Regression test accuracy:', accuracy_test)
print('\nClassification Report of Logistic Regression:')
print(metrics.classification_report(y_test, y_pred_logistictest))
print('\nConfusion Matrix of Logistic Regression:')
print(metrics.confusion_matrix(y_test, y_pred_logistictest))



#------------------------------------------------------------------
#DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

tree_classifier = DecisionTreeClassifier(random_state=42)  #ttghiar
tree_classifier.fit(X_train, y_train)

# Predictions
y_pred_decisiontrain = tree_classifier.predict(X_train)
y_pred_decisiontest = tree_classifier.predict(X_test)

# Evaluation
accuracy_train = tree_classifier.score(X_train, y_train)
accuracy_test = tree_classifier.score(X_test, y_test)
print('\nDecision Tree train accuracy:', accuracy_train)
print('Decision Tree test accuracy:', accuracy_test)
print('\nClassification Report of Decision Tree :')
print(metrics.classification_report(y_test,y_pred_decisiontest ))
print('\nConfusion Matrix of Decision Tree:')
print(metrics.confusion_matrix(y_test, y_pred_decisiontest ))

# import pickle
# with open("classification.pkl", "wb") as f:
#         pickle.dump(null_replacement, f)
#         pickle.dump(amenities_columns, f)
#         pickle.dump(pet_columns, f)
#         pickle.dump(state_mode_price, f)
#         pickle.dump(city_mode_price, f)
#         pickle.dump(cats_mode_price, f)
#         pickle.dump(dogs_mode_price, f)
#         pickle.dump(label_encoders, f)
#         pickle.dump(scaled_cols, f)
#         pickle.dump(scaler, f)
#         pickle.dump(top_feature, f)
#         pickle.dump(gb_regressor, f)

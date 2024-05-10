from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split


data_frame = pd.read_csv('ApartmentRentPrediction_Milestone2.csv')
y=data_frame['RentCategory']
X=data_frame.drop(columns=['RentCategory'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,shuffle=True,random_state=10)
train_data_frame=pd.DataFrame(X_train)
train_data_frame['RentCategory']=y_train

# train_data_frame['time'] = pd.to_datetime(train_data_frame['time'], unit='s')
# train_data_frame['time'] = pd.to_datetime(train_data_frame['time'], unit='s')
train_data_frame['currency'] = 0
train_data_frame['fee'] = 0
columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source', 'state_mode_price','city_mode_price', 'cats_mode_price','dogs_mode_price')
print(train_data_frame.columns)
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
null_replacement['pets_allowed']=""

for col in null_replacement:
    train_data_frame[col] = train_data_frame[col].fillna(null_replacement[col])

df_encoded_pets_allowed = train_data_frame['pets_allowed'].str.get_dummies(sep=',')
pet_columns=df_encoded_pets_allowed.columns.values

train_data_frame.drop(columns=['pets_allowed'], inplace=True)
train_data_frame = pd.concat([train_data_frame, df_encoded_pets_allowed], axis=1)


state_mode_price = train_data_frame.groupby('state')['RentCategory'].agg(lambda x: x.value_counts().index[0])
train_data_frame['state_mode_price'] = train_data_frame['state'].map(state_mode_price)
city_mode_price = train_data_frame.groupby('cityname')['RentCategory'].agg(lambda x: x.value_counts().index[0])
train_data_frame['city_mode_price'] = train_data_frame['cityname'].map(city_mode_price)
train_data_frame.drop(['state', 'cityname'], axis=1, inplace=True)



cats_mode_price = train_data_frame.groupby('Cats')['RentCategory'].agg(lambda x: x.value_counts().index[0])
train_data_frame['cats_mode_price'] = train_data_frame['Cats'].map(cats_mode_price)
dogs_mode_price = train_data_frame.groupby('Dogs')['RentCategory'].agg(lambda x: x.value_counts().index[0])
train_data_frame['dogs_mode_price'] = train_data_frame['Dogs'].map(dogs_mode_price)
train_data_frame.drop(['Cats', 'Dogs'], axis=1, inplace=True)

label_encoders = {} #Dictionary to store label encoders
for c in columns_for_encoding:
    lbl = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    lbl.fit(train_data_frame[[c]])
    train_data_frame[c] = lbl.transform(train_data_frame[[c]])
    label_encoders[c] = lbl



scaled_cols = list(train_data_frame.columns)
scaled_cols.remove("RentCategory")
scaler = StandardScaler()
train_data_frame[scaled_cols] = scaler.fit_transform(train_data_frame[scaled_cols])
train_data_frame.head()



# plt.scatter(train_data_frame['city_mode_price'], train_data_frame['RentCategory'])
# plt.xlabel('city_mode_price')
# plt.ylabel('RentCategory')
# plt.title('Scatter plot of city_mode_price vs RentCategory')
# plt.show()
# plt.scatter(train_data_frame['square_feet'], train_data_frame['RentCategory'])
# plt.xlabel('square_feet')
# plt.ylabel('RentCategory')
# plt.title('Scatter plot of square_feet vs RentCategory')
# plt.show()
# plt.scatter(train_data_frame['bedrooms'], train_data_frame['RentCategory'])
# plt.xlabel('bedrooms')
# plt.ylabel('RentCategory')
# plt.title('Scatter plot of bedrooms vs RentCategory')
# plt.show()


# plt.subplots(figsize=(12, 8))
# corr = train_data_frame.corr()
# top_feature = corr.index[abs(corr['RentCategory'])>0.2]
# top_corr = train_data_frame[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()

#X = train_data_frame[top_feature]


X_train = train_data_frame.drop('RentCategory', axis=1)
y_train = train_data_frame['RentCategory']

#selected features


from sklearn.feature_selection import mutual_info_classif

# Calculate Information Gain for each feature
info_gain = mutual_info_classif(X_train, y_train)

# Select the top features based on Information Gain
num_features_to_select = 20  # Change this number as per your preference
top_feature_indices = (-info_gain).argsort()[:num_features_to_select]
selected_features = X_train.columns[top_feature_indices]

print(selected_features)

X_train = X_train[selected_features]



# -------------------------------------------------

#Preprocessing test data

X_test['currency'] = 0
X_test['fee'] = 0

for col in null_replacement:
    X_test[col] = X_test[col].fillna(null_replacement[col])

X_test['amenities'] = X_test['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = X_test['amenities'].str.get_dummies(sep=',')
test_amenities_columns=df_encoded_amenities.columns.values
unseen_columns = [x for x in test_amenities_columns if x not in amenities_columns]
df_encoded_amenities.drop(columns=unseen_columns, inplace=True)
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


X_test['state_mode_price'] = X_test['state'].map(state_mode_price)
X_test['city_mode_price'] = X_test['cityname'].map(city_mode_price)
X_test.drop(['state', 'cityname'], axis=1, inplace=True)

X_test['cats_mode_price'] = X_test['Cats'].map(cats_mode_price)
X_test['dogs_mode_price'] = X_test['Dogs'].map(dogs_mode_price)
X_test.drop(['Cats', 'Dogs'], axis=1, inplace=True)

columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source', 'state_mode_price','city_mode_price', 'cats_mode_price','dogs_mode_price')
for c in columns_for_encoding:
    X_test[c] = label_encoders[c].transform(X_test[[c]])


X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])
X_test = X_test[selected_features]
#------------------------------------------------------------------------



#Random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12) #change
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

svm_classifier = SVC(kernel='rbf', random_state=42, C=0.8) #change
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

logistic_classifier = LogisticRegression(random_state=42, tol=0.0001, C=0.1) #change
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

tree_classifier = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=None)  #change
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

#handing unseen null values
null_replacement['id']=-1
null_replacement['time']=train_data_frame['time'].mean()
null_replacement['square_feet']=train_data_frame['square_feet'].mean()
null_replacement['title']=''
null_replacement['body']=''
null_replacement['price_type']=train_data_frame['price_type'].mode()
null_replacement['source']=''




import pickle
with open("classification.pkl", "wb") as f:
        pickle.dump(null_replacement, f)
        pickle.dump(amenities_columns, f)
        pickle.dump(pet_columns, f)
        pickle.dump(state_mode_price, f)
        pickle.dump(city_mode_price, f)
        pickle.dump(cats_mode_price, f)
        pickle.dump(dogs_mode_price, f)
        pickle.dump(label_encoders, f)
        pickle.dump(scaled_cols, f)
        pickle.dump(scaler, f)
        pickle.dump(selected_features, f)
        pickle.dump(rf_classifier, f)
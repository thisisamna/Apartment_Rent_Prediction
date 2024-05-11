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
train_time=0
test_time=0
start_time = time.time()

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


end_time = time.time()

# Calculate the difference
train_time += end_time - start_time


# -------------------------------------------------

start_time = time.time()

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
end_time = time.time()

test_time+= end_time - start_time

train_preprocessing_time=train_time

test_preprocessing_time=test_time
#------------------------------------------------------------------------



#Random forest 
start_time = time.time()

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12) #change
rf_classifier.fit(X_train, y_train)

end_time = time.time()
train_time+= end_time - start_time

start_time = time.time()


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
end_time = time.time()
test_time+= end_time - start_time

print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time
#------------------------------------------------------------------
#SVM
start_time = time.time()

from sklearn.svm import SVC
from sklearn import metrics

svm_classifier = SVC(kernel='rbf', random_state=42, C=0.8) #change
svm_classifier.fit(X_train, y_train)
end_time = time.time()
train_time+= end_time - start_time

start_time = time.time()

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
end_time = time.time()
test_time+= end_time - start_time

print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time
#------------------------------------------------------------------

#LOGISTIC
start_time = time.time()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logistic_classifier = LogisticRegression(random_state=42, tol=0.0001, C=0.1) #change
logistic_classifier.fit(X_train, y_train)

end_time = time.time()
train_time+= end_time - start_time
# Predictions
start_time = time.time()

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
end_time = time.time()
test_time+= end_time - start_time


print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time
#------------------------------------------------------------------
#DecisionTree
start_time = time.time()

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

tree_classifier = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=13)  #change
tree_classifier.fit(X_train, y_train)
end_time = time.time()
train_time+= end_time - start_time

# Predictions
start_time = time.time()

y_pred_decisiontrain = tree_classifier.predict(X_train)
y_pred_decisiontest = tree_classifier.predict(X_test)

#Evaluation
accuracy_train = tree_classifier.score(X_train, y_train)
accuracy_test = tree_classifier.score(X_test, y_test)
print('\nDecision Tree train accuracy:', accuracy_train)
print('Decision Tree test accuracy:', accuracy_test)
print('\nClassification Report of Decision Tree :')
print(metrics.classification_report(y_test,y_pred_decisiontest ))
print('\nConfusion Matrix of Decision Tree:')
print(metrics.confusion_matrix(y_test, y_pred_decisiontest ))
end_time = time.time()
test_time+= end_time - start_time


print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time

#------------------------------------
# Voting classifier
start_time = time.time()

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Create a voting ensemble classifier
voting_classifier = VotingClassifier(
    estimators=[('rf', rf_classifier), ('tree', tree_classifier), ('svm', svm_classifier)],
    voting='hard'
)

# Train the voting classifier
voting_classifier.fit(X_train, y_train)
end_time = time.time()
train_time+= end_time - start_time

# Predictions
start_time = time.time()

y_pred_voting_train = voting_classifier.predict(X_train)
y_pred_voting_test = voting_classifier.predict(X_test)

# Evaluation
accuracy_train = accuracy_score(y_train, y_pred_voting_train)
accuracy_test = accuracy_score(y_test, y_pred_voting_test)

print('Voting Ensemble train accuracy:', accuracy_train)
print('Voting Ensemble test accuracy:', accuracy_test)
end_time = time.time()
test_time+= end_time - start_time

print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time

#--------------------------------------------------
#KNN Classifier
start_time = time.time()

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # ttghiar

knn_classifier.fit(X_train, y_train)
end_time = time.time()
train_time+= end_time - start_time

start_time = time.time()

y_pred_knn_train = knn_classifier.predict(X_train)
y_pred_knn_test = knn_classifier.predict(X_test)

accuracy_train = knn_classifier.score(X_train, y_train)
accuracy_test = knn_classifier.score(X_test, y_test)

print('\n KNN train accuracy:', accuracy_train)
print('KNN test accuracy:', accuracy_test)
print('\nClassification Report of KNN:')
print(metrics.classification_report(y_test, y_pred_knn_test))
print('\nConfusion Matrix of KNN:')
print(metrics.confusion_matrix(y_test, y_pred_knn_test))
end_time = time.time()
test_time+= end_time - start_time

print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time

#------------------------------------------
#XGBOOST
start_time = time.time()

import xgboost as xgb
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
xgb_label_encoder = LabelEncoder()

# Fit label encoder and transform target labels
y_train_encoded = xgb_label_encoder.fit_transform(y_train)


xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train_encoded)
end_time = time.time()
train_time+= end_time - start_time

start_time = time.time()

y_test_encoded = xgb_label_encoder.transform(y_test)
y_pred_xgb_train = xgb_classifier.predict(X_train)
y_pred_xgb_test = xgb_classifier.predict(X_test)
accuracy_train = metrics.accuracy_score(y_train_encoded, y_pred_xgb_train)
accuracy_test = metrics.accuracy_score(y_test_encoded, y_pred_xgb_test)

print('\n XGBoost train accuracy:', accuracy_train)
print('XGBoost test accuracy:', accuracy_test)
print('\nClassification Report of XGBoost:')
print(metrics.classification_report(y_test_encoded, y_pred_xgb_test))
print('\nConfusion Matrix of XGBoost:')
print(metrics.confusion_matrix(y_test_encoded, y_pred_xgb_test))
end_time = time.time()
test_time+= end_time - start_time

print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time

#----------------------------------------------
#STACKING CLASSIFIER
start_time = time.time()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn import metrics

# Initialize base models
base_models = [
    ('decision_tree', DecisionTreeClassifier()),
   # ('random_forest', RandomForestClassifier()),
    ('svm', SVC(kernel='rbf', random_state=42)),
   # ('logistic', LogisticRegression()),
    ('knn', KNeighborsClassifier()),
    ('xgboost', xgb.XGBClassifier())
]

# Initialize stacking classifier with meta-model (e.g., Logistic Regression)
stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=   xgb.XGBClassifier())

# Train the stacking classifier on the training data
stacking_classifier.fit(X_train, y_train)
end_time = time.time()
train_time+= end_time - start_time

start_time = time.time()

# Predictions
y_pred_stacking_train = stacking_classifier.predict(X_train)
y_pred_stacking_test = stacking_classifier.predict(X_test)

# Evaluation
accuracy_train = metrics.accuracy_score(y_train, y_pred_stacking_train)
accuracy_test = metrics.accuracy_score(y_test, y_pred_stacking_test)

print('\n Stacking Classifier train accuracy:', accuracy_train)
print('Stacking Classifier test accuracy:', accuracy_test)
print('\nClassification Report of Stacking Classifier:')
print(metrics.classification_report(y_test, y_pred_stacking_test))
print('\nConfusion Matrix of Stacking Classifier:')
print(metrics.confusion_matrix(y_test, y_pred_stacking_test))
end_time = time.time()
test_time+= end_time - start_time

print("Total train time ",train_time," s")
print("Total test time ",test_time," s")

#reset time
train_time=train_preprocessing_time
test_time=test_preprocessing_time

#----------------------------------------------
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
        pickle.dump(svm_classifier, f)
        pickle.dump(logistic_classifier, f)
        pickle.dump(tree_classifier, f)
        pickle.dump(voting_classifier, f)
        pickle.dump(knn_classifier, f)
        pickle.dump(xgb_label_encoder,f)
        pickle.dump(xgb_classifier, f)
        pickle.dump(stacking_classifier, f)


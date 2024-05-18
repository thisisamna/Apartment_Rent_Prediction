from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import pickle

with open("classification.pkl", "rb") as f:
        null_replacement=pickle.load(f)
        amenities_columns=pickle.load(f)
        pet_columns=pickle.load(f)
        state_mode_price=pickle.load(f)
        city_mode_price=pickle.load(f)
        cats_mode_price=pickle.load(f)
        dogs_mode_price=pickle.load(f)
        label_encoders=pickle.load(f)
        scaled_cols=pickle.load(f)
        scaler=pickle.load(f)
        selected_features=pickle.load(f)
        rf_classifier=pickle.load(f)
        svm_classifier=pickle.load(f)
        logistic_classifier=pickle.load(f)
        tree_classifier=pickle.load(f)
        voting_classifier=pickle.load(f)
        knn_classifier=pickle.load(f)
        xgb_label_encoder=pickle.load(f)
        xgb_classifier=pickle.load(f)
        stacking_classifier=pickle.load(f)

data_frame = pd.read_csv('Classification_Dataset.csv')

data_frame['currency'] = 0
data_frame['fee'] = 0

for col in null_replacement:
    data_frame[col] = data_frame[col].fillna(null_replacement[col])

data_frame['amenities'] = data_frame['amenities'].str.replace(r'[/\s]', ',')
df_encoded_amenities = data_frame['amenities'].str.get_dummies(sep=',')
test_amenities_columns=df_encoded_amenities.columns.values
unseen_columns = [x for x in test_amenities_columns if x not in amenities_columns]
df_encoded_amenities.drop(columns=unseen_columns, inplace=True)
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


data_frame['state_mode_price'] = data_frame['state'].map(state_mode_price)
data_frame['city_mode_price'] = data_frame['cityname'].map(city_mode_price)
data_frame.drop(['state', 'cityname'], axis=1, inplace=True)

data_frame['cats_mode_price'] = data_frame['Cats'].map(cats_mode_price)
data_frame['dogs_mode_price'] = data_frame['Dogs'].map(dogs_mode_price)
data_frame.drop(['Cats', 'Dogs'], axis=1, inplace=True)

columns_for_encoding = ('category', 'title', 'body', 'has_photo', 'price_type', 'address', 'source', 'state_mode_price','city_mode_price', 'cats_mode_price','dogs_mode_price')
for c in columns_for_encoding:
    data_frame[c] = label_encoders[c].transform(data_frame[[c]])


data_frame[scaled_cols] = scaler.transform(data_frame[scaled_cols])

X = data_frame[selected_features]
y = data_frame['RentCategory']

#Random forest 
# Predictions
ypred_RF= rf_classifier.predict(X)

# Evaluation
accuracy = rf_classifier.score(X, y)
print('\nRandom Forest accuracy:', accuracy)
print('\nClassification Report of Random Forest:')
print(metrics.classification_report(y, ypred_RF))
print('\nConfusion Matrix of Random Forest:')
print(metrics.confusion_matrix(y, ypred_RF))


#------------------------------------------------------------------

# Predictions
ypred_svm = svm_classifier.predict(X)

# Evaluation
accuracy = svm_classifier.score(X, y)
print('\n SVM  accuracy:', accuracy)
print('\nClassification Report of SVM:')
print(metrics.classification_report(y, ypred_svm))
print('\nConfusion Matrix of SVM:')
print(metrics.confusion_matrix(y, ypred_svm))


#------------------------------------------------------------------

#LOGISTIC

# Predictions

ypred_logistic = logistic_classifier.predict(X)

# Evaluation
accuracy = logistic_classifier.score(X, y)
print('\nLogistic Regression  accuarcy:', accuracy)
print('\nClassification Report of Logistic Regression:')
print(metrics.classification_report(y, ypred_logistic))
print('\nConfusion Matrix of Logistic Regression:')
print(metrics.confusion_matrix(y, ypred_logistic))


#------------------------------------------------------------------
#DecisionTree

# Predictions
ypred_decision = tree_classifier.predict(X)
ypred_decision = tree_classifier.predict(X)

#Evaluation
accuracy = tree_classifier.score(X, y)
print('\nDecision Tree  accuracy:', accuracy)
print('\nClassification Report of Decision Tree :')
print(metrics.classification_report(y,ypred_decision ))
print('\nConfusion Matrix of Decision Tree:')
print(metrics.confusion_matrix(y, ypred_decision ))



#------------------------------------
# Voting classifier
from sklearn.metrics import accuracy_score

# Predictions


y_pred_voting = voting_classifier.predict(X)

# Evaluation
accuracy = accuracy_score(y, y_pred_voting)

print('Voting Ensemble accuracy:', accuracy)



#--------------------------------------------------
#KNN Classifier

y_pred_knn = knn_classifier.predict(X)
accuracy = knn_classifier.score(X, y)

print('\n KNN accuracy:', accuracy)
print('\nClassification Report of KNN:')
print(metrics.classification_report(y, y_pred_knn))
print('\nConfusion Matrix of KNN:')
print(metrics.confusion_matrix(y, y_pred_knn))



#------------------------------------------
#XGBOOST


y_encoded = xgb_label_encoder.transform(y)
y_pred_xgb = xgb_classifier.predict(X)
accuracy = metrics.accuracy_score(y_encoded, y_pred_xgb)
accuracy = metrics.accuracy_score(y_encoded, y_pred_xgb)

print('\n XGBoost accuracy:', accuracy)
print('\nClassification Report of XGBoost:')
print(metrics.classification_report(y_encoded, y_pred_xgb))
print('\nConfusion Matrix of XGBoost:')
print(metrics.confusion_matrix(y_encoded, y_pred_xgb))



#----------------------------------------------
#STACKING CLASSIFIER


# Predictions
y_pred_stacking = stacking_classifier.predict(X)

# Evaluation
accuracy = metrics.accuracy_score(y, y_pred_stacking)

print('\n Stacking Classifier accuracy:', accuracy)
print('\nClassification Report of Stacking Classifier:')
print(metrics.classification_report(y, y_pred_stacking))
print('\nConfusion Matrix of Stacking Classifier:')
print(metrics.confusion_matrix(y, y_pred_stacking))



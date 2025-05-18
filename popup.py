import pandas as pd
data= pd.read_csv('popup_dataset_3000.csv')
df= pd.DataFrame(data)
df.head()
from sklearn.preprocessing import LabelEncoder
features= {}
for column in ['Type', 'Size', 'Trigger', 'Design', 'Has_CTA', 'Position', 'Content_Type']:
    le= LabelEncoder()
    df[column]= le.fit_transform(df[column])
    features[column]= le
target_dictionary= {}
target= LabelEncoder()
df['Category']= target.fit_transform(df['Category'])
target_dictionary['Category']= target
df.head()
from sklearn.model_selection import train_test_split
x= df.iloc[:,:-1]
y= df.iloc[:, -1]
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.3, random_state= 42)
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(multi_class= 'multinomial', solver='lbfgs', max_iter=1000)
lr.fit(x_train, y_train)
y_pred= lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy= accuracy_score(y_test, y_pred)
print(accuracy)
cf= confusion_matrix(y_test, y_pred)
print(cf)
cr= classification_report(y_test, y_pred)
print(cr)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt= dt.predict(x_test)
accuracy_dt= accuracy_score(y_test, y_pred_dt)
print("Accuracy Score for Decision Trees:\n", accuracy_dt)
cf_dt= confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix for Decision Trees: \n", cf_dt)
cr_dt= classification_report(y_test, y_pred_dt)
print(cr_dt)
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf= rf.predict(x_test)
accuracy_rf= accuracy_score(y_test, y_pred_rf)
print("Accuracy Score for Random Forests:\n", accuracy_rf)
cf_rf= confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix for Random Forests:\n", cf_rf)
cr_rf= classification_report(y_test, y_pred_rf)
print("Classification Report for Random Forests:\n", cf_rf)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=3000)
mlp.fit(x_train, y_train)
y_pred_mlp = mlp.predict(x_test)
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

import numpy as np

def predict_category(user_input, rf_model, encoders, target_encoder):
    feature_names = ['Type', 'Size', 'Trigger', 'Design', 'Has_CTA', 'Position', 'Content_Type']
    
    encoded_input = []
    for feature in feature_names:
        if feature in encoders:
            encoded_value = encoders[feature].transform([user_input[feature]])[0]
            encoded_input.append(encoded_value)
        else:
            raise ValueError(f"Missing encoder for feature: {feature}")

    input_df = pd.DataFrame([encoded_input], columns=feature_names)

    predicted_label = rf_model.predict(input_df)[0]
    decoded_output = target_encoder.inverse_transform([predicted_label])[0]

    return decoded_output

def runtime_user_input_predict(rf_model, encoders, target_encoder):
    feature_names = ['Type', 'Size', 'Trigger', 'Design', 'Has_CTA', 'Position', 'Content_Type']
    user_input = {}

    for feature in feature_names:
        options = list(encoders[feature].classes_)
        print(f"\nEnter {feature} (Options: {options})")
        while True:
            value = input(f"{feature}: ").strip()
            if value in options:
                user_input[feature] = value
                break
            else:
                print(f"Invalid input. Please enter one of: {options}")

    # Encode input
    encoded_input = [encoders[feature].transform([user_input[feature]])[0] for feature in feature_names]
    input_df = pd.DataFrame([encoded_input], columns=feature_names)

    # Predict and decode
    predicted_label = rf_model.predict(input_df)[0]
    decoded_output = target_encoder.inverse_transform([predicted_label])[0]

    print(f"\n✅ Predicted Category: {decoded_output}")

# 👇 Call this function instead of static prediction
runtime_user_input_predict(rf, features, target)



import joblib

# Save the RandomForest model
joblib.dump(rf, 'popup_category_rf_model.joblib')

# Save the features LabelEncoders dictionary
joblib.dump(features, 'features_encoders.joblib')

# Save the target LabelEncoder
joblib.dump(target, 'target_encoder.joblib')

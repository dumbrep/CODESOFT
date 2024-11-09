#Loding necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#loading training dataset
data = pd.read_csv("fraudTrain.csv")

data = data.drop(["Unnamed: 0","trans_num","cc_num","first","last","street","lat","long","unix_time"],axis = 1)

#preprocessing training dataset
encoder = LabelEncoder()
data["gender"] = encoder.fit_transform(data["gender"])


category_encoder = OneHotEncoder()
encoded_category = category_encoder.fit_transform(data[["category"]]).toarray()
category_names = category_encoder.get_feature_names_out()
category_df = pd.DataFrame(encoded_category, columns=category_names)

data = pd.concat([data.drop("category", axis = 1), category_df], axis=1)

data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
data['transaction_hour'] = data['trans_date_trans_time'].dt.hour
data['transaction_day'] = data['trans_date_trans_time'].dt.day
data['transaction_month'] = data['trans_date_trans_time'].dt.month
data['transaction_weekday'] = data['trans_date_trans_time'].dt.weekday 
data.drop("trans_date_trans_time",axis = 1, inplace=True)
frequency = data['merchant'].value_counts()
data['merchant'] = data['merchant'].map(frequency)


frequency = data['city'].value_counts()
data['city'] = data['city'].map(frequency)



frequency = data['state'].value_counts()
data['state'] = data['state'].map(frequency)
data.columns

data["dob"] = pd.to_datetime(data["dob"])
def calculate_age(birthdate):
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


data['age'] = data['dob'].apply(calculate_age)

data.drop("dob",axis = 1, inplace = True)

job_encoder = LabelEncoder()
data["job"] = job_encoder.fit_transform(data["job"])

len(data.columns)
input = data.iloc[:,[i for i in range(0,30) if i != 10]]
output = data.iloc[:,10]

#creating Machine Learning Models
tree_model = DecisionTreeClassifier()
tree_model.fit(input,output)
regression_model = LogisticRegression()
regression_model.fit(input,output)
data_test = pd.read_csv("fraudTest.csv")



#importing testing dataset
data_test = data_test.drop(["Unnamed: 0","trans_num","cc_num","first","last","street","lat","long","unix_time"],axis = 1)

encoder = LabelEncoder()
data_test["gender"] = encoder.fit_transform(data_test["gender"])

category_encoder = OneHotEncoder()
encoded_category = category_encoder.fit_transform(data_test[["category"]]).toarray()
category_names = category_encoder.get_feature_names_out()
category_df = pd.DataFrame(encoded_category, columns=category_names)

data_test = pd.concat([data_test.drop("category", axis = 1), category_df], axis=1)

data_test["trans_date_trans_time"] = pd.to_datetime(data_test["trans_date_trans_time"])
data_test['transaction_hour'] = data_test['trans_date_trans_time'].dt.hour
data_test['transaction_day'] = data_test['trans_date_trans_time'].dt.day
data_test['transaction_month'] = data_test['trans_date_trans_time'].dt.month
data_test['transaction_weekday'] = data_test['trans_date_trans_time'].dt.weekday 
data_test.drop("trans_date_trans_time",axis = 1, inplace=True)

frequency = data_test['merchant'].value_counts()
data_test['merchant'] = data_test['merchant'].map(frequency)


frequency = data_test['city'].value_counts()
data_test['city'] = data_test['city'].map(frequency)


frequency = data_test['state'].value_counts()
data_test['state'] = data_test['state'].map(frequency)


data_test["dob"] = pd.to_datetime(data_test["dob"])
def calculate_age(birthdate):
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


data_test['age'] = data_test['dob'].apply(calculate_age)

data_test.drop("dob",axis = 1, inplace = True)

job_encoder = LabelEncoder()
data_test["job"] = job_encoder.fit_transform(data_test["job"])

test_input = data_test.iloc[:,[i for i in range(0,30) if i != 10]]

#predicting output
tree_predicted = tree_model.predict(test_input)
regression_predicted = regression_model.predict(test_input)

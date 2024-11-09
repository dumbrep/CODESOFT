#importing necessary labries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

#loading dataframe
data = pd.read_csv("Churn_Modelling.csv")
data.drop(["RowNumber","CustomerId","Surname"], axis = 1,inplace = True)

#preprocessing the input data
gender_encoder = LabelEncoder()
data["Gender"] = gender_encoder.fit_transform(data["Gender"])
geo_encoder = OneHotEncoder()
encoded_geo = geo_encoder.fit_transform(data[["Geography"]]).toarray()
geo_names = geo_encoder.get_feature_names_out()
geo_df = pd.DataFrame(encoded_geo, columns=geo_names)
data = pd.concat([data.drop("Geography", axis = 1), geo_df], axis=1)

#preparing input and output
x = data.iloc[:,[i for i in range(0,13) if i!= 9]]
y = data.iloc[:,9]

#seperating training and testing data
x_train, x_test, y_train,y_test = tts(x, y, test_size=0.2)

#building and training models
model1 = LogisticRegression()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model1.fit(x_train,y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
predict1 = model1.predict(x_test)
predict2 = model2.predict(x_test)
predict3= model3.predict(x_test)
accuracy1 = accuracy_score(y_test, predict1)
accuracy2 = accuracy_score(y_test, predict2)
accuracy3 = accuracy_score(y_test, predict3)

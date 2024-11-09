#import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

#loading dataset
data = pd.read_csv("spam.csv",sep = "\t", names = ["result", "message"])

#preprocessing dataset
result_encoder = LabelEncoder()
data["result"] = result_encoder.fit_transform(data["result"])
msg_encoder = TfidfVectorizer(stop_words="english")
x = msg_encoder.fit_transform(data["message"]).toarray()
y = data["result"]

#splitting dataset into training and testing dataset
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2)

#building models
model1 = MultinomialNB()
model2 = LogisticRegression()
model3 = SVC()
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
predicted1 = model1.predict(x_test)
predicted2 = model2.predict(x_test)
predicted3 = model3.predict(x_test)

#evaluating models
accuracy_1 = accuracy_score(predicted1,y_test)
accuracy_2 = accuracy_score(predicted2,y_test)
accuracy_3 = accuracy_score(predicted3,y_test)

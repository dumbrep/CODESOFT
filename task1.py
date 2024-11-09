#importing libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

#loading train data
data = pd.read_csv("train_data.txt",sep = ":::",header = None)


#cheking is there any null value present
data[data.isna().any(axis = 1)]

#convering Title and Description into vectors using TFiDF
embeder = TfidfVectorizer(stop_words="english",max_features=1000)
x1= embeder.fit_transform(data[1]).toarray()


embeder2 = TfidfVectorizer(stop_words="english",max_features=1000)
x2 = embeder2.fit_transform(data[3]).toarray()

x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)

#setting up inputs and outputs
inputs = pd.concat([x1,x2], axis= 1)
y = data[2]

#loading test data
test_data = pd.read_csv("test_data.txt", sep = ":::", header = None)
test_data
test_x1 = embeder.transform(test_data[1]).toarray()


##convering Title and Description of test data into vectors using TFiDF
test_x2 = embeder2.transform(test_data[2]).toarray()
test_x1 = pd.DataFrame(test_x1)
test_x2 = pd.DataFrame(test_x2)
test_inputs = pd.concat([test_x1,test_x2],axis = 1)


#building Navie Bayes model 
model = MultinomialNB()
model.fit(inputs, y)
predicted = model.predict(test_inputs)

#building Logistic regression model
model_logistic = LogisticRegression()
model_logistic.fit(inputs,y)
predicted_logistic = model_logistic.predict(test_inputs)

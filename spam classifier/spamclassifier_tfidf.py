# importing dataset
import pandas as pd
messages = pd.read_csv("smsspamcollection",sep='\t', names=["label","message"])

# data cleaning and preprocessing
import re
import nltk
nltk.download("stopwords")
nltk.download('wordnet')

from nltk.corpus import stopwords

# using stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    msg = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    msg = msg.lower()
    msg = msg.split()
    msg = [ps.stem(word) for word in msg if not word in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg)

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages["label"])
y = y.iloc[:,1].values

# test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
 
# naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB()
spam_detect.fit(X_train,y_train)

# predicting
y_pred = spam_detect.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
# accuracy = 0.9739910313901345

# using lemmatization
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
corpus = []
for i in range(len(messages)):
    msg = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    msg = msg.lower()
    msg = msg.split()
    msg = [wnl.lemmatize(word) for word in msg if not word in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg)

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages["label"])
y = y.iloc[:,1].values

# test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
 
# naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB()
spam_detect.fit(X_train,y_train)

# predicting
y_pred = spam_detect.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
# accuracy =0.9766816143497757
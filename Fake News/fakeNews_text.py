# importing dataset
import pandas as pd
df = pd.read_csv('dataset/train.csv')
df.head(5)

# independent variable
X = df.drop('label', axis=1) 
X.head(5)

# dependent variable
y = df['label']

# droping missing values
df = df.dropna()
df.shape()

messages = df.copy()

# reset index
messages.reset_index(inplace=True)

#cleaning data
import regex as re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    msg = re.sub('[^a-zA-Z]',' ',messages['text'][i])
    msg = msg.lower()
    msg = msg.split()
    msg  = [ps.stem(word) for word in msg if not word in stopwords.words('english')]
    msg = ' '.join(msg)
    corpus.append(msg)

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
y = messages['label']

# test train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# naive bayes
from sklearn.naive_bayes import MultinomialNB
classifier_nb = MultinomialNB()
classifier_nb.fit(X_train, y_train)

# predicting
y_pred = classifier_nb.predict(X_test)

# confusion metric, accuracy, report
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm = confusion_matrix(y_test,y_pred)
cm

accuracy = accuracy_score(y_test,y_pred)
accuracy
#accuracy = 0.8990976210008204

report = classification_report(y_test,y_pred)
print(report)


# random forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0, verbose=2)
classifier_rf.fit(X_train, y_train)

# predicting
y_pred = classifier_rf.predict(X_test)

# confusion metric, accuracy, report
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm = confusion_matrix(y_test,y_pred)
cm

accuracy = accuracy_score(y_test,y_pred)
accuracy
#accuracy = 0.9458572600492207

report = classification_report(y_test,y_pred)
print(report)


# passive aggressive classifier
from sklearn.linear_model import PassiveAggressiveClassifier
classifier_pa = PassiveAggressiveClassifier(n_iter_no_change = 50)
classifier_pa.fit(X_train, y_train)

# predicting
y_pred = classifier_pa.predict(X_test)

# confusion metric, accuracy, report
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm = confusion_matrix(y_test,y_pred)
cm

accuracy = accuracy_score(y_test,y_pred)
accuracy
#accuracy = 0.9450369155045119

report = classification_report(y_test,y_pred)
print(report)


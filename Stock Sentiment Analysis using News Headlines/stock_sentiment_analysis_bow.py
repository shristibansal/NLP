# importing dataset
import pandas as pd
df = pd.read_csv("Stock_Data.csv", encoding="ISO-8859-1")
df.head(5)

# splitting dataset on the basis of date
train = df[df["Date"] < '20150101']
test = df[df['Date'] > '20141231']

# data cleaning
# removing puncuation
data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

# renaming columns
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index
data.head(5)

# 
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
headlines[0]

# bow model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(2,2))
train_data = cv.fit_transform(headlines)

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
classifier.fit(train_data, train['Label'])

# prediction
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_data = cv.transform(test_transform)
pred = classifier.predict(test_data)

# confusion metric, accuracy, report
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm = confusion_matrix(test['Label'], pred)
cm
accuracy = accuracy_score(test['Label'], pred)
accuracy
#accuracy = 0.8544973544973545
report = classification_report(test['Label'], pred)
report
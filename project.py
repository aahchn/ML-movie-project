#Project.py
#Prabhmeet Gill
#Aaron Chan
#Tarun Salh

import pandas as pd
import csv
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction import text


feature_data = []
sentiment_data =[]


##### Reading in the train set

with open('train.csv', 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    next(lines)
	    for row in lines:
	    	feature_data.append((row[2]))
	    	sentiment_data.append((row[3]))



feature_test_data = []
test_phrase_id = []

##### Reading in the test set

with open(sys.argv[1], 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    next(lines)
	    for row in lines:
	    	test_phrase_id.append((row[0]))
	    	feature_test_data.append((row[2]))



##### Using bag of words
#common_stop_words = text.ENGLISH_STOP_WORDS
#stop_words = common_stop_words
vectorizer = CountVectorizer()
count_Vector = vectorizer.fit_transform(feature_data)

#scaled_train_data = preprocessing.scale(count_Vector.toarray())


##### Using the transformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(count_Vector.toarray())


##### Set up test data for algorithim

count_test_Vector = vectorizer.transform(feature_test_data)
#scaled_test_data = preprocessing.scale(count_test_Vector.toarray())
test_tfidf = transformer.fit_transform(count_test_Vector.toarray())



##### implementing naivebayes using the sklearn library
#naive_bayes = MultinomialNB()
#naive_bayes.fit(tfidf, sentiment_data)

#output_naive_bayes = naive_bayes.predict(test_tfidf)
#accuracy of 0.6302521008403361%

###### implementing One vs All
#output_OnevsAll = OneVsRestClassifier(LinearSVC(random_state=0)).fit(tfidf,sentiment_data).predict(test_tfidf)
#accuracy of 0.7292891012614195%

##### implementing One vs One
output_OnevsOne = OneVsOneClassifier(LinearSVC(random_state=0)).fit(tfidf,sentiment_data).predict(test_tfidf)
#accuracy of 0.7456930484612145%

print(output_OnevsOne)

##### record the accuracy, precision, recall, and score

accuracy = accuracy_score(sentiment_data, output_OnevsOne)
precision = precision_score(sentiment_data, output_OnevsOne, average='macro')
recall = recall_score(sentiment_data, output_OnevsOne, average='macro')
F1_score = 2 * (precision * recall) / (precision + recall)


print(accuracy)
print(precision)
print(recall)
print(F1_score)


##### output to csv

csv_output = pd.DataFrame(data={"PhraseId":test_phrase_id,"Sentiment":output_OnevsOne})
csv_output.to_csv("PrabhmeetGill_TarunSalh_AaronChan_predictions.csv",index=False,quoting=3)


print(csv_output)

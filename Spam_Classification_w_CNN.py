import math
import re
import string
import time
from collections import OrderedDict
from configparser import ConfigParser
from itertools import combinations
import scipy.sparse as sp
import matplotlib.pyplot as plt
import nltk
from matplotlib.colors import ListedColormap

import tensorflow as tf
from nltk.corpus import stopwords
from IPython.display import display
import numpy as np
import seaborn as sns
from sklearn import metrics
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, \
    f1_score,confusion_matrix

from keras.preprocessing import text
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfTransformer

from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D, Dropout, BatchNormalization

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

stemmer = nltk.SnowballStemmer("english")


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords


def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text


def stemm_text(text):
    stemmer = nltk.SnowballStemmer("english")
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))

    return text


def main():
    file_name = "spam.csv"

    inputFile = 'data/' + file_name

    df = pd.read_csv(inputFile, encoding="ISO-8859-1")
    df.head()

    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

    df.rename(columns={"v1": "Target", "v2": "Text"}, inplace=True)

    cols = ["#E0D298", "#6C6491"]
    # first of all let us evaluate the target and find out if our data is imbalanced or not
    plt.figure(figsize=(12, 8))
    fg = sns.countplot(x=df["Target"], palette=cols)
    fg.set_title("Count Plot of Classes", color="#58508d")
    fg.set_xlabel("Classes", color="#58508d")
    fg.set_ylabel("Number of Data points", color="#58508d")

    labels = df.iloc[:, 0].to_numpy()

    encoder = LabelEncoder()

    labels = encoder.fit_transform(labels)

    df['message_clean'] = df['Text'].apply(preprocess_data)

    X = df['message_clean']

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test

    vect = CountVectorizer()
    vect.fit(X_train)

    x_train_dtm = vect.transform(X_train)
    x_test_dtm = vect.transform(X_test)

    vocab_size = 1000
    maxlen = 1000
    embedding_dims = 50
    epochs = 20
    VOCAB_SIZE = len(vect.vocabulary_)

    # Padded data for CNN
    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train2 = tokenizer.texts_to_sequences(X_train)
    X_test2 = tokenizer.texts_to_sequences(X_test)

    X_train2 = sequence.pad_sequences(X_train2, maxlen=maxlen)
    X_test2 = sequence.pad_sequences(X_test2, maxlen=maxlen)

    # Start creating the CNN model
    CNNmodel = Sequential()

    CNNmodel.add(Embedding(VOCAB_SIZE, 50, input_length=maxlen))

    CNNmodel.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1))
    CNNmodel.add(GlobalMaxPooling1D())

    CNNmodel.add(Dropout(0.5))
    CNNmodel.add(BatchNormalization())
    CNNmodel.add(Dropout(0.5))

    CNNmodel.add(Dense(256, activation='relu'))

    CNNmodel.add(Dropout(0.5))
    CNNmodel.add(BatchNormalization())
    CNNmodel.add(Dropout(0.5))

    CNNmodel.add(Dense(1, activation='sigmoid'))
    CNNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    CNNmodel.summary()
    print("Success!")

    # fit the data
    history = CNNmodel.fit(X_train2, y_train,
                           epochs=epochs,
                           batch_size=128,
                           validation_data=(X_test2, y_test))

    # evaluate the model
    _, train_acc = CNNmodel.evaluate(X_train2, y_train, verbose=0)
    _, test_acc = CNNmodel.evaluate(X_test2, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    CNN_train_accuracy = CNNmodel.evaluate(X_train2, y_train)
    CNN_test_accuracy = CNNmodel.evaluate(X_test2, y_test)

    yhat_probs = CNNmodel.predict(X_test2, verbose=0)
    # predict crisp classes for test set
    yhat_classes = yhat_probs.round()

    CNNprec = metrics.precision_score(y_test, yhat_classes)
    CNNrecal = metrics.recall_score(y_test, yhat_classes)
    CNNf1_s = metrics.f1_score(y_test, yhat_classes)

    # classifiers -> SVM , NB,  Decision Tree
    classifiers = [SVC(),
                   MultinomialNB(),
                   DecisionTreeClassifier()
                   ]

    for cls in classifiers:
        cls.fit(x_train_dtm, y_train)

    pipe_dict = {0: "SVM", 1: "Multinomial NB", 2: "Decision Tree"}

    for i, model in enumerate(classifiers):
        cv_score = cross_val_score(model, x_train_dtm, y_train, scoring="accuracy", cv=10)
        print("%s: %f " % (pipe_dict[i], cv_score.mean()))

    precision = []
    recall = []
    f1_score = []
    trainset_accuracy = []
    testset_accuracy = []

    for i in classifiers:
        pred_test = i.predict(x_test_dtm)
        prec = metrics.precision_score(y_test, pred_test)
        recal = metrics.recall_score(y_test, pred_test)
        f1_s = metrics.f1_score(y_test, pred_test)
        train_accuracy = model.score(x_train_dtm, y_train)
        test_accuracy = model.score(x_test_dtm, y_test)

        # Appending scores
        precision.append(prec)
        recall.append(recal)
        f1_score.append(f1_s)
        trainset_accuracy.append(train_accuracy)
        testset_accuracy.append(test_accuracy)

    # append cnn scores
    precision.append(CNNprec)
    recall.append(CNNrecal)
    f1_score.append(CNNf1_s)
    trainset_accuracy.append(CNN_train_accuracy[1])
    testset_accuracy.append(CNN_test_accuracy[1])

    data = {'Precision': precision,
            'Recall': recall,
            'F1score': f1_score,
            'Accuracy on Testset': testset_accuracy,
            'Accuracy on Trainset': trainset_accuracy}

    # Creates pandas DataFrame.
    Results = pd.DataFrame(data, index=["SVM", "Multinomial NB", "Decision Tree", "CNN"])

    Results.to_excel("output.xlsx")

    cmap = ListedColormap(["#6B4226", "#EBC79E"])
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for cls, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(cls,
                              x_test_dtm,
                              y_test,
                              ax=ax,
                              cmap=cmap,
                              )
        ax.title.set_text(type(cls).__name__)
        # 0 -> ham 1 -> spam
        ax.xaxis.set_ticklabels(['ham', 'spam'])
        ax.yaxis.set_ticklabels(['ham', 'spam'])

    plt.tight_layout()
    plt.show()

    CNN_cm = confusion_matrix(y_test, yhat_classes)

    # after creating the confusion matrix, for better understanding plot the cm.

    plt.figure(figsize=(10, 7))
    sns.heatmap(CNN_cm, annot=True,fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

    return 0


if __name__ == '__main__':
    main()

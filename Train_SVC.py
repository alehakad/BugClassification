## for data
import pandas as pd
import numpy as np

## language recognition
from langdetect import detect

## for processing
import re
import nltk



## for bag-of-words
from sklearn import (
    feature_extraction,
    model_selection,
    pipeline,
    feature_selection,
    metrics,
    svm,
)


# данные из jira
data_jira = pd.read_csv("data_from_jira.csv")
data_jira.shape

# удаление строк с проектами demo, integrate
data_jira = data_jira[
    (data_jira.company != "DEMO") & (data_jira.company != "INTEGRATE")
]

# определение языков писем
text_langs = list(map(lambda x: detect(x) == "en", data_jira.text))


# удаление строк не с английским языком
data_jira_en = data_jira[text_langs]

# распределение компаний
n_companies_sorted = (
    data_jira_en["company"]
    .reset_index()
    .groupby("company")
    .count()
    .sort_values(by="index")
)


# удаление строк с 1 значениями
data_jira_en = data_jira_en[
    data_jira_en["company"].map(lambda x: n_companies_sorted.loc[x][0] > 1)
]


ps = nltk.stem.porter.PorterStemmer()
lem = nltk.stem.wordnet.WordNetLemmatizer()
# препроцессинг текста
def preprocess_text(
    text,
    flg_stemm=False,
    flg_lemm=True,
    lst_stopwords=nltk.corpus.stopwords.words("english"),
):
    # текст переводится в нижний регистр, оставляются только буквы
    text = re.sub("[^A-Za-z ]+", "", text.lower().strip())

    # токенизация - разбиение на слова
    lst_text = text.split()

    # удаление стоп слов
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # стемминг
    if flg_stemm:

        lst_text = [ps.stem(word) for word in lst_text]

    # леммантизация
    if flg_lemm:

        lst_text = [lem.lemmatize(word) for word in lst_text]

    # перевод обратно в текст
    text = " ".join(lst_text)
    return text


# стоп слова
lst_stopwords = nltk.corpus.stopwords.words("english")


data_jira_en["text_clean"] = data_jira_en["text"].apply(
    lambda x: preprocess_text(
        x, flg_stemm=True, flg_lemm=False, lst_stopwords=lst_stopwords
    )
)


## split dataset
dtf_train, dtf_test = model_selection.train_test_split(
    data_jira_en, test_size=0.33, shuffle=True, stratify=data_jira_en["company"]
)
## get target
y_train = dtf_train["company"].values
y_test = dtf_test["company"].values

corpus = dtf_train["text_clean"]

## Tf-Idf (advanced variant of BoW)
vectorizer = feature_extraction.text.TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2)
)

vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

# отбор признаков для уменьшения размерности матрицы с помощью хи-квадрат

y = dtf_train["company"]
X_names = vectorizer.get_feature_names()
p_value_limit = 0.0001
dtf_features = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X_train, y == cat)
    dtf_features = dtf_features.append(
        pd.DataFrame({"feature": X_names, "score": p, "y": cat})
    )
    dtf_features = dtf_features.sort_values(["y", "score"], ascending=[True, False])
    dtf_features = dtf_features[dtf_features["score"] < p_value_limit]
X_names = dtf_features["feature"].unique().tolist()

vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_


# SVM classifier
X_test = dtf_test["text_clean"].values

classifier_svm = svm.SVC(kernel="linear", probability=True)

## pipeline
model_svm = pipeline.Pipeline(
    [("vectorizer", vectorizer), ("classifier", classifier_svm)]
)


# fit the training dataset on the classifier
model_svm["classifier"].fit(X_train, y_train)
# predict the labels on validation dataset
predicted_svm = model_svm.predict(X_test)
# Use accuracy_score function to get the accuracy
print(predicted_svm)


## Accuracy, Precision, Recall
accuracy = metrics.accuracy_score(y_test, predicted_svm)

print("Accuracy:", round(accuracy, 2))
print("Detail:")
print(metrics.classification_report(y_test, predicted_svm))

# сохранение обученной модели
import pickle

Pkl_Filename = "model_svm.pkl"

with open(Pkl_Filename, "wb") as file:
    pickle.dump(model_svm, file)

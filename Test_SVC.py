import pickle
import re
import nltk
import sys
from googletrans import Translator


def preprocess_text(
    text,
    flg_stemm=False,
    flg_lemm=True,
    lst_stopwords=nltk.corpus.stopwords.words("english"),
):
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()

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


if len(sys.argv) == 3:
    file, model_filename, email_filename = sys.argv

    # считывание письма из файла
    with open(email_filename, "rb") as file:
        email = file.read()

    # считывание обученной модели из файла
    with open(model_filename, "rb") as file:
        model_SVM = pickle.load(file)

    # препроцессинг текста

    translator = Translator()

    test_email_en = translator.translate(email).text

    pre_text = preprocess_text(test_email_en, flg_stemm=True, flg_lemm=False)

    # ответ
    ans = model_SVM.predict([pre_text])[0]

    # запись ответа в файл
    with open("company_name.txt", "w") as f:
        f.write(ans)

    print("Answer file created")

else:
    print("Wrong input")

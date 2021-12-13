import string
import re
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk import word_tokenize, download
from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem


def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])


def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])


def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)


def remove_symbols(df_res):
    print('\nФильтрация цифр и знаков препинания')
    return [remove_multiple_spaces(
        remove_numbers(
            remove_punctuation(text.lower())
        )
    ) for text in tqdm(df_res['text'])]


def remove_stop_words(prep_text):
    print('\nФильтрация "стопслов"')
    # download('stopwords')
    # download('punkt')
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(['…', '«', '»', '...', '–'])
    sw_texts_list = []
    for text in tqdm(prep_text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in russian_stopwords and token != ' ']
        text = " ".join(tokens)
        sw_texts_list.append(text)
    return sw_texts_list


def stemming(prep_text):
    print('\nСтэмминг')
    stemmer = SnowballStemmer("russian")
    stemmed_texts_list = []
    for text in tqdm(prep_text):
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        text = " ".join(stemmed_tokens)
        stemmed_texts_list.append(text)
    return stemmed_texts_list


def lemming(prep_text):
    print('\nЛемматизация')
    mystem = Mystem()
    lemm_texts_list = []
    for text in tqdm(prep_text):
        text_lem = mystem.lemmatize(text)
        tokens = [token for token in text_lem if token != ' ']
        text = " ".join(tokens)
        lemm_texts_list.append(text)
    return lemm_texts_list


def filter_data_texts(df_res):
    return lemming(
        stemming(
            remove_stop_words(
                remove_symbols(df_res)
            )
        )
    )

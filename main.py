import pandas as pd
from tqdm.auto import tqdm
from filter_data_texts import filter_data_texts
from analysis import analysis


def getNewsByTopics(data):
    print('\nПолучение данных по топикам')
    newsCountByTopic = 250
    topicList = data['topic'].unique()
    df_res = pd.DataFrame()
    for topic in tqdm(topicList):
        df_topic = data[data['topic'] == topic][:newsCountByTopic]
        df_res = df_res.append(df_topic, ignore_index=True)
    return df_res


def prepare_df_res():
    print('\nРаспаковка данных для обучения')
    df_res = getNewsByTopics(pd.read_csv('dataset/lenta-ru-train.csv'))
    df_res['text_lemm'] = filter_data_texts(df_res)
    df_res.to_csv('dataset/lemm.csv')
    return df_res


def get_df_res():
    try:
        print('\nРаспаковка готовых данных для обучения')
        df_res = pd.read_csv('dataset/lemm.csv', encoding='utf-8')
    except FileNotFoundError:
        print('\nГотовые данные не сохранены, подготовка данных')
        df_res = prepare_df_res()
    return df_res


def prepare_df_test():
    testCount = 100
    print('\nРаспаковка тестовых данных')
    df_test = pd.read_csv('dataset/lenta-ru-test.csv')[:testCount]
    df_test['text_lemm'] = filter_data_texts(df_test)
    df_test.to_csv('dataset/lemm-test.csv')
    return df_test


def get_df_test():
    try:
        print('\nРаспаковка готовых тестовых данных')
        df_test = pd.read_csv('dataset/lemm-test.csv', encoding='utf-8')
    except FileNotFoundError:
        print('\nГотовые данные не сохранены, подготовка данных')
        df_test = prepare_df_test()
    print(df_test['text'][0], df_test['text_lemm'][0])
    return df_test


def main():
    analysis(get_df_res(), get_df_test())


if __name__ == '__main__':
    main()

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


def analysis(df_res, df_test):
    x = df_res['text_lemm']
    y = df_res['topic']

    nb = Pipeline([
        ('vect', CountVectorizer()),  # векторизация текста в матрицу по частотам (частота встречаемости токенов)
        ('tfidf', TfidfTransformer()),  # рассчет ОБРАТНОЙ частоты встречаемости токена (контекстные "стопслова")
        ('clf', MultinomialNB())  # наивный байесовский классификатор
    ])
    nb.fit(x, y)
    prediction = nb.predict(df_test['text_lemm'])
    df_test['prediction'] = prediction
    print()
    for row in df_test.values.tolist():
        print('{0}) {1} - {2}'.format(row[0], row[1], row[4]))

    # print('\n', classification_report(y[:20], prediction, target_names=df_res['topic'].unique()))
    # print(df_test[['title', 'prediction']])

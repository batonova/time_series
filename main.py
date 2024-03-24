import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import json

from prophet import Prophet
import tsfresh
from tsfresh import extract_relevant_features
from tsfresh import extract_features

if __name__ == '__main__':
    # load data
    train = pd.read_csv("train.csv")
    train.rename(columns={'дата': 'date', 'направление': 'class', 'выход': 'out'}, inplace=True)
    train.date = pd.to_datetime(train.date, dayfirst=True)
    train.out = train.out.str.replace(',', '.').astype(float)

    test = pd.read_csv("test.csv")
    test.rename(columns={'дата': 'date', 'направление': 'class', 'выход': 'out'}, inplace=True)
    test.date = pd.to_datetime(test.date, dayfirst=True)

    # transform data for Prophet
    train_prophet = train.drop(['class'], axis=1)
    train_prophet.rename(columns={'out': 'y', 'date': 'ds'}, inplace=True)
    test_prophet = test.drop(['class', 'out'], axis=1)
    test_prophet.rename(columns={'date': 'ds'}, inplace=True)
    test_prophet = pd.concat([test_prophet, train_prophet.drop(['y'], axis=1)])

    # train
    model = Prophet()
    model.fit(train_prophet)

    # Predict regression
    forecast = model.predict(test_prophet)
    y_reg_pred = list(forecast[1001:]['yhat'])

    base = train.sort_values(by='date').set_index(train.index)
    base = base.replace({'л': 1, 'ш': 0})
    base = pd.concat([base, test.set_index(test.index + 1001)])

    columns = ['id', 'date', 'out', 'y']
    features = pd.DataFrame(columns=columns)
    columns_y = ['y']
    target = pd.DataFrame(columns=columns_y)

    window = 6
    for index, row in base.iterrows():

        for i in range(index - window + 1, index + 1):
            if i >= 0:
                features.loc[len(features.index)] = [index + 1, base.iloc[i]['date'], base.iloc[i]['out'], row['class']]
        target.loc[len(target.index) + 1] = [row['class']]

    features = features.drop('y', axis=1)
    target = features.drop(['date', 'out'], axis=1).drop_duplicates()

    features_filtered_direct = extract_features(features, column_id='id', column_sort='date', ml_task='classification')

    X_train, X_test, y_train = features_filtered_direct[:1001], features_filtered_direct[1001:], target[:1001]
    cl = DecisionTreeClassifier()
    cl.fit(X_train, y_train)

    y_class_pred = list(cl.predict(X_test))

    with open('forecast_class.json', 'w') as file:

        json.dump(y_class_pred, file)

    with open('forecast_value.json', 'w') as file:

        json.dump(y_reg_pred, file)





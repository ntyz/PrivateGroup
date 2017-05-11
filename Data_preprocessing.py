# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split


def Data_preprocessing(directory='../pre/', test_size=0.2):

    # Preprocess data, read csv file, split train set and test set.

    train_file_name = 'train.csv'

    df_train = pd.read_csv(directory + train_file_name)
    df_train.clickTime = df_train.clickTime % 10000 // 100

    df_user = pd.read_csv(directory + 'user.csv')
    df_useri = pd.read_csv(directory + 'user_installedapps.csv')
    # df_useraa = pd.read_csv(directory + 'user_app_actions.csv')
    df_ad = pd.read_csv(directory + 'ad.csv')
    df_appc = pd.read_csv(directory + 'app_categories.csv')
    df_pos = pd.read_csv(directory + 'position.csv')

    df_ad_app = pd.merge(df_ad, df_appc, on="appID", suffixes=('_a', '_b'))
    df_train_user = pd.merge(df_train, df_user, on="userID", suffixes=('_a', '_b'))
    df_train_user_app = pd.merge(df_train_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
    df_train_user_app_pos = pd.merge(df_train_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))
    df_train_all = pd.merge(df_train_user_app_pos,df_useri,on = 'userID',suffixes=('_a', '_b'))

    # df_test = pd.read_csv(test_file_name)
    # df_test_user = pd.merge(df_test, df_user, on="userID", suffixes=('_a', '_b'))
    # df_test_user_app = pd.merge(df_test_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
    # df_test_all = pd.merge(df_test_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))

    train_all_x = df_train_all.drop(['label', 'conversionTime'], axis=1)
    train_all_y = df_train_all.ix[:, ['label', 'conversionTime']]  # Two columns, label and conversionTime

    x_train, x_test, y_train, y_test = train_test_split(train_all_x, train_all_y, test_size=test_size, random_state=42)

    df_train_all.to_csv(directory + "train_all.csv", index=None)
    train_all_x.to_csv(directory + 'train_all_x.csv', index=None)
    train_all_y.to_csv(directory + 'train_all_y.csv', index=None)
    # df_test_all.to_csv(directory + "test_all.csv", index=None)

    return x_train, x_test, y_train, y_test

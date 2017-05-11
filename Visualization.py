# -*- coding:utf-8 -*-

# import wide_deep
import pandas as pd

directory = '../pre/'
train_file_name = directory + 'train.csv'
test_file_name = directory + 'test.csv'

print("data processing...")
# 特征整合
df_train = pd.read_csv(train_file_name)
df_train.clickTime = df_train.clickTime % 10000//100
df_test = pd.read_csv(test_file_name)

df_user = pd.read_csv(directory+'user.csv')
df_ad = pd.read_csv(directory+'ad.csv')
df_app = pd.read_csv(directory+'app_categories.csv')
df_pos = pd.read_csv(directory+'position.csv')

df_ad_app = pd.merge(df_ad, df_app, on="appID", suffixes=('_a', '_b'))

df_train_user = pd.merge(df_train, df_user, on="userID", suffixes=('_a', '_b'))
df_train_user_app = pd.merge(df_train_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
df_train_all = pd.merge(df_train_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))

df_test_user = pd.merge(df_test, df_user, on="userID", suffixes=('_a', '_b'))
df_test_user_app = pd.merge(df_test_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
df_test_all = pd.merge(df_test_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))
df_train_all.to_csv(directory+"train_all.csv",index=None)
df_test_all.to_csv(directory+"test_all.csv",index=None)
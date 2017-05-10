# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import pandas as pd
import tensorflow as tf
import os
from sklearn.utils import shuffle
import scipy as sp
from tensorflow.contrib.learn import LinearClassifier,DNNClassifier,DNNLinearCombinedClassifier
from tensorflow.contrib.layers import embedding_column, sparse_column_with_integerized_feature
SUB = True
directory = "pre/"
model_dir="model"
model_type="wide_n_deep"
train_steps=200

COLUMNS = [u'label', u'clickTime', u'conversionTime', u'creativeID', u'userID',
       u'positionID', u'connectionType', u'telecomsOperator', u'age',
       u'gender', u'education', u'marriageStatus', u'haveBaby', u'hometown',
       u'residence', u'adID', u'camgaignID', u'advertiserID', u'appID',
       u'appPlatform', u'appCategory']
#没用conversionTime creativeID userID appID
CATEGORICAL_COLUMNS = ["clickTime",
                       "positionID", "connectionType", "telecomsOperator",'age',
                       'gender', 'education', 'marriageStatus','haveBaby','hometown',
                       'residence','adID','camgaignID','advertiserID',
                       'appPlatform','appCategory']
CONTINUOUS_COLUMNS = []

#构建模型
def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  clickTime = tf.contrib.layers.sparse_column_with_integerized_feature(
      "clickTime", bucket_size=24)
  # creativeID = tf.contrib.layers.sparse_column_with_integerized_feature(
  #     "creativeID", bucket_size=7000)
  positionID = tf.contrib.layers.sparse_column_with_integerized_feature(
      "positionID", bucket_size=7646)
  connectionType = tf.contrib.layers.sparse_column_with_integerized_feature(
      "connectionType", bucket_size=5)
  telecomsOperator = tf.contrib.layers.sparse_column_with_integerized_feature(
      "telecomsOperator", bucket_size=4)
  age = tf.contrib.layers.sparse_column_with_integerized_feature(
      "age", bucket_size=81)
  gender =tf.contrib.layers.sparse_column_with_integerized_feature(
      "gender", bucket_size=3)
  education = tf.contrib.layers.sparse_column_with_integerized_feature(
      "education", bucket_size=8)
  marriageStatus = tf.contrib.layers.sparse_column_with_integerized_feature(
      "marriageStatus", bucket_size=4)
  haveBaby= tf.contrib.layers.sparse_column_with_integerized_feature(
      "haveBaby", bucket_size=7)
  hometown= tf.contrib.layers.sparse_column_with_integerized_feature(
      "hometown", bucket_size=365)
  residence= tf.contrib.layers.sparse_column_with_integerized_feature(
      "residence", bucket_size=400)
  adID= tf.contrib.layers.sparse_column_with_integerized_feature(
      "adID", bucket_size=3616)
  camgaignID=tf.contrib.layers.sparse_column_with_integerized_feature(
      "camgaignID", bucket_size=720)
  advertiserID=tf.contrib.layers.sparse_column_with_integerized_feature(
      "advertiserID", bucket_size=91)
  appPlatform=tf.contrib.layers.sparse_column_with_integerized_feature(
      "appPlatform", bucket_size=3)
  appCategory=tf.contrib.layers.sparse_column_with_integerized_feature(
      "appCategory", bucket_size=504)
  wide_columns = [ clickTime,  positionID, connectionType,
                   telecomsOperator,age,gender,education,marriageStatus,haveBaby,
                   hometown,residence,adID,camgaignID,advertiserID,appPlatform,appCategory,
                  # tf.contrib.layers.crossed_column([education, occupation],
                  #                                  hash_bucket_size=int(1e4)),
                  # tf.contrib.layers.crossed_column(
                  #     [age_buckets, education, occupation],
                  #     hash_bucket_size=int(1e6)),
                   tf.contrib.layers.crossed_column([clickTime, connectionType,telecomsOperator],
                                                    hash_bucket_size=int(1e4))
                ]
  deep_columns = [
      tf.contrib.layers.embedding_column(clickTime, dimension=8),
      tf.contrib.layers.embedding_column(positionID, dimension=8),
      tf.contrib.layers.embedding_column(connectionType, dimension=8),
      tf.contrib.layers.embedding_column(telecomsOperator,
                                         dimension=8),
      tf.contrib.layers.embedding_column(age, dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(marriageStatus, dimension=8),
      tf.contrib.layers.embedding_column(haveBaby, dimension=8),
      tf.contrib.layers.embedding_column(hometown, dimension=8),
      tf.contrib.layers.embedding_column(residence, dimension=8),
      tf.contrib.layers.embedding_column(adID, dimension=8),
      tf.contrib.layers.embedding_column(camgaignID, dimension=8),
      tf.contrib.layers.embedding_column(advertiserID, dimension=8),
      tf.contrib.layers.embedding_column(appCategory, dimension=8),
      tf.contrib.layers.embedding_column(appPlatform, dimension=8)
  ]
  if model_type == "wide":
    m = LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
  return m

def input_fn(df):
  """Input builder function."""

  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  feature_cols = dict(categorical_cols)

  # Converts the label column into a constant Tensor.
  label = tf.constant(df["label"].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def data_process(train_file_name, test_file_name):
    print("data processing...")
    # 特征整合
    df_train = pd.read_csv(train_file_name)
    df_train.clickTime = df_train.clickTime%10000//100
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
    #数据划分0.8
    train = shuffle(df_train_all)
    n = int(train['label'].count() * 0.8)
    train_div = train[:n]
    test_div = train[n:]
    train_div.to_csv(directory+"train_div.csv",index=None)
    test_div.to_csv(directory+"test_div.csv",index=None)
    return 0

#评分函数
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = -sp.mean(act*sp.log(pred) + sp.subtract(1,act)*sp.log(1-pred))
  return ll

#模型训练和预测

train_data=directory+"train.csv"
test_data=directory+"test.csv"
if not (os.path.exists(directory+"train_div.csv") or os.path.exists(directory+"test_div.csv")):
    data_process(train_data, test_data)
if(SUB):
    print("use all train data")
    df_train = pd.read_csv(directory+"train_all.csv")
    df_test = pd.read_csv(directory+"test_all.csv")
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir, model_type)
    x_train, y_train =input_fn(df_train)
    m.fit(x=x_train,y=y_train, steps=train_steps, batch_size=256)
    x_test, y_test=input_fn(df_test)
    pred = m.predict_proba(x = x_test, y=y_test, as_iterable=False, batch_size=256)
    pred = pred[:, 1]

    # 输出
    df_test['prob'] = pred
    ans = df_test[['instanceID', 'prob']]
    ans.to_csv(directory + 'submission.csv', index=None)
else:
    print("use only 80% train data")
    df_train = pd.read_csv(directory+"train_div.csv")
    df_test = pd.read_csv(directory+"test_div.csv")
    df_train = pd.read_csv(directory+"train_all.csv")
    df_test = pd.read_csv(directory+"test_all.csv")
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir, model_type)
    x_train, y_train =input_fn(df_train)
    m.fit(x=x_train,y=y_train, steps=train_steps, batch_size=256)
    x_test, y_test=input_fn(df_test)
    results = m.evaluate(x = x_test, y=y_test, as_iterable=False, batch_size=256)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

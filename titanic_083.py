#!/usr/bin/env python
# coding: utf-8
# train score 0.8316498316498316
# kaggle score 0.78468

import warnings

<<<<<<< HEAD
import numpy as np
import pandas as pd
=======
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection
>>>>>>> faec7c90a94c9cab491e16051eee2a63e02a2dff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

<<<<<<< HEAD

def main():
    warnings.filterwarnings('ignore')

    # CSVを読み込む
    print('read csv')
    train = pd.read_csv("input_csv/train.csv")
    test = pd.read_csv("input_csv/test.csv")

    # データの統合
    print('merge train test')
    dataset = pd.concat([train, test], ignore_index=True)

    # 提出用に
    PassengerId = test['PassengerId']

    # Cabin は一旦除外
    del dataset["Cabin"]

    # Age(年齢)とFare(料金)はそれぞれの中央値、Embarked(出港地)はS(Southampton)を代入
    print('filllna')
    dataset["Age"].fillna(dataset.Age.mean(), inplace=True) 
    dataset["Fare"].fillna(dataset.Fare.mean(), inplace=True) 
    dataset["Embarked"].fillna("S", inplace=True)

    # 使用する変数を抽出
    dataset = dataset[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp']]

    # ダミー変数を作成
    print('make dummy data')
    dataset_dummies = pd.get_dummies(dataset)

    # データをtrainとtestに分解 
    train_set = dataset_dummies[0:train.shape[0]]
    test_set = dataset_dummies[train.shape[0]:]
    del test_set["Survived"]

    # trainデータを変数と正解に分離
    X = train_set.copy()
    del X["Survived"]
    y = train_set.Survived # 正解データ

    # 予測モデルの作成
    print('modeling')
    clf = RandomForestClassifier(random_state=10, max_features='sqrt')
    pipe = Pipeline([('classify', clf)])
    param_test = {'classify__n_estimators': list(range(20, 30, 1)),  # 20～30を１刻みずつ試す
                  'classify__max_depth': list(range(3, 10, 1))}      # 3～10を１刻みずつ試す
    grid = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='accuracy', cv=10)

    grid.fit(X, y)
    print(grid.best_params_, grid.best_score_, sep="\n")

    # testデータの予測
    pred = grid.predict(test_set)

    # Kaggle提出用csvファイルの作成
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred.astype(np.int32)})
    #submission.to_csv("submission2.csv", index=False)
    print('finish')


if __name__ == '__main__':
    main()
=======
warnings.filterwarnings('ignore')

# CSVを読み込む
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# データの統合
dataset = pd.concat([train, test], ignore_index=True)

# 提出用に
PassengerId = test['PassengerId']

# 全体の欠損データの個数確認
dataset_null = dataset.fillna(np.nan)
dataset_null.isnull().sum()

# Cabin は一旦除外
del dataset["Cabin"]

# Age(年齢)とFare(料金)はそれぞれの中央値、Embarked(出港地)はS(Southampton)を代入
dataset["Age"].fillna(dataset.Age.mean(), inplace=True) 
dataset["Fare"].fillna(dataset.Fare.mean(), inplace=True) 
dataset["Embarked"].fillna("S", inplace=True)

# 使用する変数を抽出
dataset2 = dataset[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp']]

# ダミー変数を作成
dataset_dummies = pd.get_dummies(dataset2)

# データをtrainとtestに分解 
train_set = dataset_dummies[dataset_dummies['Survived'].notnull()]
test_set = dataset_dummies[dataset_dummies['Survived'].isnull()]
del test_set["Survived"]

# trainデータを変数と正解に分離
X = train_set.as_matrix()[:, 1:] # Pclass以降の変数
y = train_set.as_matrix()[:, 0]  # 正解データ

# 予測モデルの作成
clf = RandomForestClassifier(random_state=10, max_features='sqrt')
pipe = Pipeline([('classify', clf)])
param_test = {'classify__n_estimators': list(range(20, 30, 1)),  # 20～30を１刻みずつ試す
              'classify__max_depth': list(range(3, 10, 1))}      # 3～10を１刻みずつ試す
grid = GridSearchCV(estimator = pipe, param_grid=param_test, scoring='accuracy', cv=10)
grid.fit(X, y)
print(grid.best_params_, grid.best_score_, sep="\n")

# testデータの予測
pred = grid.predict(test_set)

# Kaggle提出用csvファイルの作成
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred.astype(np.int32)})
#submission.to_csv("submission2.csv", index=False)
>>>>>>> faec7c90a94c9cab491e16051eee2a63e02a2dff

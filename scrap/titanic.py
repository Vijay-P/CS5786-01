#!/usr/bin/env python3

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import pandas as pd

def write_result(result):
    with open('result.csv', 'w+') as f:
        f.write("PassengerId,Survived\n")
        for row in result:
            f.write(" %d,%d\n" % row)

classifier = LogisticRegression(C=1,
                                penalty='l2',
                                tol=0.01)

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

dropped_columns = ['PassengerId',
                   'Name',
                   'Ticket',
                   'Fare',
                   'Cabin',
                   'Embarked']

df_train = pd.read_csv("../titanic_data/train.csv")
df_test = pd.read_csv("../titanic_data/test.csv")

df_test_features = df_test.drop(dropped_columns,axis=1)
df_train_features = df_train.drop(dropped_columns + ["Survived"], axis=1)

df_train_survived = df_train.iloc[:,1]

df_train_features = pd.get_dummies(df_train_features)
df_test_features = pd.get_dummies(df_test_features)

df_test_idx = df_test.iloc[:,0]

df_train_features = imputer.fit_transform(df_train_features)
df_test_features = imputer.fit_transform(df_test_features)

classifier.fit(df_train_features, df_train_survived)
classifier.score(df_train_features, df_train_survived)

classifier.predict(df_test_features)
result_labels = classifier.predict(df_test_features)

ys = zip(df_test_idx,result_labels)

write_result(ys)


print('Done!')

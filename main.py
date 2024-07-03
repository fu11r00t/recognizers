import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
print(os.listdir("./"))
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
y_train = train_df['label'].values.flatten()
x_train = train_df.drop(['label'],axis=1).values
x_test=test_df
transformer = MinMaxScaler().fit(x_train,y_train)
x_train = transformer.transform(x_train)
x_test = transformer.transform(x_test)
def train_predict():
    from sklearn.svm import SVC
    clf=SVC(C=10, kernel="rbf")
    clf.fit(x_train,y_train)
    return clf.predict(x_test)
y_pred_original = train_predict()
sub=pd.read_csv('./sample_submission.csv')
sub.Label = y_pred_original
sub.to_csv('submission.csv',index=False)
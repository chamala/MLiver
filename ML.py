import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")

df=pd.read_csv("data.csv")



Y =df["Dataset"]

df =df.drop("Dataset",axis=1)

df_2 = pd.get_dummies(df, columns=["Gender"], drop_first=False)



X = df_2.values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
df = pd.DataFrame(x_scaled)


df.fillna(df.mean(), inplace=True)

df = np.array(df)
Y = Y.astype('int')
df = df.astype('int')


X_train, X_test, Y_train, Y_test = train_test_split(df,Y, test_size=0.30, random_state=42)



clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, Y_train)


pickle.dump(clf, open('model.pkl', 'wb'))
model=pickle.load(open('model.pkl','rb'))
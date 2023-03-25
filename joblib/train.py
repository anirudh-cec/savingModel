import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
df= pd.read_csv(url,names=names)
X=df.iloc[:,0:8]
Y=df.iloc[:,8]
print(df)
print(X)
print(Y)

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.20,random_state=101)
model=LogisticRegression()
model.fit(X_train,Y_train)
print('[info] model has been trained')

result = model.score(X_test,Y_test)
print(f'accuracy of the model is {result}')

joblib.dump(model,'dib_70.pkl')
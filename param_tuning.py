from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
d=load_iris()
#print(d)
x=pd.DataFrame(d.data)
y=pd.DataFrame(d.target)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report
p={'n_estimators':[2,10,100],'max_depth':[3,5]}

g=GridSearchCV(param_grid=p,estimator=RandomForestClassifier())
g.fit(X_train,y_train)
y_pred=g.predict(X_test)
print(accuracy_score(y_test,y_pred))

from sklearn.datasets import make_classification
x,y=make_classification(n_samples=10,n_features=4)
print(x.shape)
print(y.shape)
print(x)
print(y)
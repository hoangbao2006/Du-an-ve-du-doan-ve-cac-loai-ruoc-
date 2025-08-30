import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

wine=load_wine()
x=pd.DataFrame(wine.data,columns=wine.feature_names)
y=wine.target
print ("Thong tin dataset")
print (x.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit (x_train,y_train)

y_pred=model.predict(x_test)

print ("Do chinh xac:",accuracy_score(y_test,y_pred))
print ("Bao cao ket qua:",classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title ("Confusion Matrix-Wine Dataset")
plt.show ()
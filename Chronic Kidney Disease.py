import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
# Importing the data
data = pd.read_csv(r'D:/Data science Notes/Python/Projects/Chronic Kidney Disease - Classification/kidney_disease.csv')
data=data.drop(columns='id',axis=1)

# Data cleaning 

data[['sg','al','su']]=data[['sg','al','su']].astype(object)
data['pcv']=pd.to_numeric(data['pcv'],errors='coerce')
data['wc']=pd.to_numeric(data['wc'],errors='coerce')
data['rc']=pd.to_numeric(data['rc'],errors='coerce')

print(data.head(20))
print(data.info())


catcol=[]
for i in data.columns:
    if data[i].dtype=="object":
        catcol.append(i)        
print(catcol)

for col in catcol:
    print(col,": ",data[col].unique())

data['dm']=data['dm'].replace({' yes':'yes','\tyes':'yes','\tno':'no'})   # we have 3 types of yes which we will convert to one value 'yes'
data['cad']=data['cad'].replace('\tno','no')
data['classification']=data['classification'].replace('ckd\t','ckd')

data['classification']=data['classification'].map({'ckd':1,'notckd':0})
data['classification']=pd.to_numeric(data['classification'],errors='coerce')


num_col=[]
for i in data.columns:
    if (data[i].dtype=="float64" or data[i].dtype=="int64"):
        num_col.append(i)        
print(num_col)
#print(data.info())

corr=data[num_col].corr()
sns.heatmap(corr,annot=True)
# plt.show()


# Filling null values with random sampling 
def random_sampling(feature):
    random_sample=data[feature].dropna().sample(data[feature].isna().sum()) # dropping nulll values and taking the same count of samples
    random_sample.index=data[data[feature].isnull()].index    # taking the index for random sample
    data.loc[data[feature].isnull(),feature]=random_sample   # storing the random sample in place of null values
print(data[num_col].isnull().sum().sort_values(ascending=False))

# replacing null vales with random sampling    
for i in num_col:
    random_sampling(i)                      # we can also replace it with mean if there are low counts of null values
print(data[num_col].isnull().sum())

def replace_with_mode(feature):
    mode=data[feature].mode()[0]
    data[feature]=data[feature].fillna(mode)

for i in catcol:
    replace_with_mode(i)

print(data.isnull().sum())
print(catcol)
print(data.head(20))

# Encoding the data for cat cols
le=LabelEncoder()
data[catcol]=data[catcol].apply(le.fit_transform)

X=data.drop(columns='classification',axis=1)
y=data['classification']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=24)


Model1=LogisticRegression(random_state=24)
Model1.fit(X_train,y_train)
y_pred=Model1.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model2=KNeighborsClassifier()
Model2.fit(X_train,y_train)
y_pred=Model2.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model3=GaussianNB()
Model3.fit(X_train,y_train)
y_pred=Model3.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model4=MultinomialNB()
Model4.fit(X_train,y_train)
y_pred=Model4.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model5=SVC(kernel='poly')
Model5.fit(X_train,y_train)
y_pred=Model5.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model6=KNeighborsClassifier()
Model6.fit(X_train,y_train)
y_pred=Model6.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model7=DecisionTreeClassifier()
Model7.fit(X_train,y_train)
y_pred=Model7.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model8=RandomForestClassifier()
Model8.fit(X_train,y_train)
y_pred=Model8.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

Model9=AdaBoostClassifier()
Model9.fit(X_train,y_train)
y_pred=Model9.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

Model10=XGBClassifier()
Model10.fit(X_train,y_train)
y_pred=Model10.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


models = {'xgb': Model10, 'rf': Model8,'ab':Model9}

param_grids = {
    'xgb': {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.1, 0.01]
    },
    'rf': {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [5, 10, 15],
        'rf__min_samples_split': [2, 5, 10]
    },
    'ab': {
        'ab__n_estimators': [100, 200, 300],
        
    }
}

pipelines = []

for model_name, model in models.items():
    pipeline = Pipeline([(model_name, model)])
    pipelines.append((model_name, pipeline))

best_models = {}
for model_name, pipeline in pipelines:
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grids[model_name], 
                               cv=5, 
                               scoring='accuracy') 
    grid_search.fit(X_train, y_train) 
    best_models[model_name] = {'model': grid_search.best_estimator_, 
                              'best_params': grid_search.best_params_, 
                              'best_score': grid_search.best_score_}

best_model_name = max(best_models, key=lambda x: best_models[x]['best_score'])
best_model = best_models[best_model_name]['model']
best_params = best_models[best_model_name]['best_params']

print(f"Best Model: {best_model_name}")
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_models[best_model_name]['best_score']}")

finalmodel=RandomForestClassifier(n_estimators=300,min_samples_split=5,max_depth=5)
Model1.fit(X_train,y_train)
y_pred=Model1.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

from keras.src.models import Sequential
from keras.src.layers import Dense

model=Sequential()
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=100)
y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

print(X_train.head())

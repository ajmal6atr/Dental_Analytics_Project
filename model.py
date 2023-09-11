import pandas as pd
import numpy as np
from matplotlib import pyplot 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("df.csv")

df = df.drop(['Appointment_Unique_ID'],axis = 1)
df = df.drop(['Unnamed: 0'],axis = 1)
df = df.drop(['Unnamed: 0.1'],axis = 1)
df = df.drop(['Patient_Unique_ID'],axis = 1)

### AGE ###

df.loc[(df['Age']>=3) & (df['Age']<=10), 'age'] = '3-10'
df.loc[(df['Age']>10) & (df['Age']<=30), 'age'] = '11-30'
df.loc[(df['Age']>30) & (df['Age']<=60), 'age'] = '31-60'
df.loc[(df['Age']>60), 'age'] = '>60'

### DISTANCE ###

df.loc[(df['Distance']<=4), 'distance'] = '1-4'
df.loc[(df['Distance']>4) & (df['Distance']<=7), 'distance'] = '5-7'
df.loc[(df['Distance']>=8) & (df['Distance']<=12), 'distance'] = '8-12'

### MAPPING
df[['Employment_status']] = df[['Employment_status']].replace(to_replace = {'Employed':1,'Unemployed':0})
df[['Insurance']] = df[['Insurance']].replace(to_replace = {'Yes':1,'No':0})
df[['Show up']] = df[['Show up']].replace(to_replace = {'YES':1,'NO':0})
df[['Gender']] = df[['Gender']].replace(to_replace = {'F':0,'M':1})

#pie chart showing percentage of 'Show up' and 'No Show'

Show_up = len(df[df['Show up'] == 1])
No_show = len(df[df['Show up'] == 0])
plt.pie(x = [Show_up,No_show],explode = (0,0),labels = ['Show_up','No_show'],autopct = '%1.2f%%',shadow = False,startangle = 90)
plt.show()

#converting data type to categorical
df['Employment_status'] = df['Employment_status'].astype('category')
df['Insurance'] = df['Insurance'].astype('category')
df['Show up'] = df['Show up'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['Type_of_Treatment'] = df['Type_of_Treatment'].astype('category')
df['Appointment Day'] = df['Appointment Day'].astype('category')

df = df.drop(['Age'],axis = 1)
df = df.drop(['Distance'],axis = 1)

df = pd.get_dummies(df,columns = ["Appointment Day",'Type_of_Treatment',"Cost_of_Treatment","age","distance"])
df.iloc[:,4:] = df.iloc[:,4:].astype('category')


from sklearn.feature_selection import mutual_info_classif
Xf = df.iloc[:,df.columns != 'Show up']
Yf = df.loc[:,'Show up']

MI = mutual_info_classif(Xf,Yf,random_state = 1)
featurescore = pd.DataFrame(MI,index = Xf.columns,columns=['scores'])
mutualtop = featurescore.sort_values(by =['scores'],ascending=False)
mutualtop.plot.bar(legend = None, rot=0, figsize=(28,7))
multitop = mutualtop.head(10)
multitop.plot.bar(legend=None, rot=0, figsize=(28,5))


X = Xf[multitop.index] #  top 10 features
y = Yf   

from sklearn.model_selection import train_test_split
import time 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Gradientbooster
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(max_depth=4)
t1 = time.time()
gb.fit(X_train, y_train)
pred4 = gb.predict(X_test)
t2 = time.time()
accuracy_score(y_test, pred4)*100 # test accuracy
trn4 = accuracy_score(y_train, gb.predict(X_train))*100
trn4  # train accuracy

# Saving model to disk
pickle.dump(gb, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0,1,1,1,0,0,0,1,0,0]]))

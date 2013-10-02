import pandas as pd
import numpy as np
import statsmodels.api as sm

import statsmodels.nonparametric
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/guagong/Kaggle/Titanic/lib")
import kaggleaux as ka

df = pd.read_csv("train.csv")
print df

df = df.drop(['Ticket','Cabin'], axis = 1)
df = df.dropna()
print df

##--general plot
fig = plt.figure(figsize=(18, 6))
a = 0.2
a_bar = 0.55

plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts().plot(kind='bar',alpha=a_bar)
plt.title("Distribution of Survival, (1=Survived)")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived,df.Age,alpha=a)
plt.ylabel("Age")
plt.grid(b=True,which='Major',axis='y')
plt.title("Survial by Age, (1=Survived)")

plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts().plot(kind="bar",alpha=a_bar)
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
df.Age[df.Pclass == 1].plot(kind='kde')   
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")                         
plt.title("Age Distribution within classes"); 
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts().plot(kind='bar', alpha=a_bar)
plt.title("Passengers per boarding location")
plt.show()

##--general survivors
plt.figure(figsize=(6,4))
df.Survived.value_counts().plot(kind='barh', color = "blue", alpha = .65)
plt.title("Survival Breakdown (1=Survived, 0=Died)")
plt.show()

##---gender survivors
fig = plt.figure(figsize=(18,6))

fig.add_subplot(121)
df.Survived[df.Sex=='male'].value_counts().plot(kind='barh',label='Male')
df.Survived[df.Sex=='female'].value_counts().plot(kind='barh',color='#FA2379',label='Female')
plt.title("Who Survived? with respect to Gender, (raw value counts)")
plt.legend(loc='best')

fig.add_subplot(122)
(df.Survived[df.Sex == 'male'].value_counts()/float(df.Sex[df.Sex == 'male'].size)).plot(kind='barh',label='Male')  
(df.Survived[df.Sex == 'female'].value_counts()/float(df.Sex[df.Sex == 'female'].size)).plot(kind='barh', color='#FA2379',label='Female')
plt.title("Who Survived proportionally? with respect to Gender")
plt.legend(loc='best')
plt.show()

##--Pclass
fig=plt.figure(figsize=(18,4))
a=.65 # our alpha or opacity level.

# building on the previous code, here we create an additional subset with in the gender subset we created for the survived variable. 
# I know, thats a lot of subsets. After we do that we call value_counts() so it it can be easily plotted as a bar graph. 
# this is repeated for each gender class pair.
ax1=fig.add_subplot(141)
df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts().plot(kind='bar', label='female highclass', color='#FA2479', alpha=a)
ax1.set_xticklabels(["Survived", "Died"], rotation=0)
plt.title("Who Survived? with respect to Gender and Class"); 
plt.legend(loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink', alpha=a)
ax2.set_xticklabels(["Died","Survived"], rotation=0)
plt.legend(loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts().plot(kind='bar', label='male, low class',color='lightblue', alpha=a)
ax3.set_xticklabels(["Died","Survived"], rotation=0)
plt.legend(loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts().plot(kind='bar', label='male highclass', alpha=a, color='steelblue')
ax4.set_xticklabels(["Died","Survived"], rotation=0)
plt.legend(loc='best')
plt.show()


# model formula
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)' # here the ~ sign is an = sign, and the features of our dataset
                                                                       # are written as a formula to predict survived. The C() lets our 
                                                                       # regression know that those variables are categorical.
results = {} # create a results dictionary to hold our regression results for easy analysis later



# create a regression freindly dataframe using patsy's dmatrices function
y,x = dmatrices(formula, data=df, return_type='dataframe')

# instantiate our model
model = sm.Logit(y,x)

# fit our model to the training data
res = model.fit()

# save the result for outputing predictions later
results['Logit'] = [res, formula]
print res.summary()


##--plot of result of logit
plt.figure(figsize=(18,4));
plt.subplot(121,axisbg="#DBDBDB")
ypred = res.predict(x)
plt.plot(x.index,ypred,'bo',x.index,y,'mo',alpha=.25);
plt.grid(color='white',linestyle='dashed')
plt.title('Logit predictions, Blue: \nFitted/predicted values: Red');

plt.subplot(122,axisbg="#DBDBDB")
plt.plot(res.resid, 'r-')
plt.grid(color='white',linestyle='dashed')
plt.title('Logit Residuals');
plt.show()

##--check if how the result behaves

fig = plt.figure(figsize=(18,9), dpi=1600)
a = .2

# Below are examples of more advanced plotting. 
# It it looks strange check out the tutorial above.
fig.add_subplot(221, axisbg="#DBDBDB")
kde_res = statsmodels.nonparametric.kde(res.predict())
kde_res.fit()
plt.plot(kde_res.support,kde_res.density)
plt.fill_between(kde_res.support,kde_res.density, alpha=a)
title("Distribution of our Predictions")

fig.add_subplot(222, axisbg="#DBDBDB")
plt.scatter(res.predict(),x['C(sex)[T.male]'] , alpha=a)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted chance of survival")
plt.ylabel("Gender Bool")
title("The Change of Survival Probability by Gender (1 = Male)")

fig.add_subplot(223, axisbg="#DBDBDB")
plt.scatter(res.predict(),x['C(pclass)[T.3]'] , alpha=a)
plt.xlabel("Predicted chance of survival")
plt.ylabel("Class Bool")
plt.grid(b=True, which='major', axis='x')
title("The Change of Survival Probability by Lower Class (1 = 3rd Class)")

fig.add_subplot(224, axisbg="#DBDBDB")
plt.scatter(res.predict(),x.age , alpha=a)
plt.grid(True, linewidth=0.15)
title("The Change of Survival Probability by Age")
plt.xlabel("Predicted chance of survival")
plt.ylabel("Age")
plt.show()

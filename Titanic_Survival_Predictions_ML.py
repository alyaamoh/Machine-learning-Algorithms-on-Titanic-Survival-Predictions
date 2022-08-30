
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import normalize 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn import datasets


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1


# In[2]:


Data = pd.read_csv('train.csv')
Data.head()


# In[3]:


Data.drop(['PassengerId','Name','Embarked'], axis = 1, inplace = True)
Data.head(10)


# In[4]:


Data.drop(['Ticket','Fare','Cabin'], axis = 1, inplace = True)
Data.head(10)


# In[5]:


Sex_coder = {'male':1, 'female':0}
Data.Sex = Data.Sex.map(Sex_coder)
Data.head()


# In[6]:


Survived = Data.Survived
Data.drop('Survived', axis = 1, inplace = True)
Data['Survived'] = Survived
Data = Data[pd.notnull(Data['Age'])]
Data = Data[pd.notnull(Data['Sex'])]
Data = Data[pd.notnull(Data['Parch'])]
Data = Data[pd.notnull(Data['SibSp'])]
Data = Data[pd.notnull(Data['Pclass'])]
Data = Data[pd.notnull(Data['Survived'])]
Data.head()


# In[7]:


Data.replace([np.inf, -np.inf], 0, inplace=True)
Data.groupby('Survived').mean()


# In[8]:


Data_d = Data[Data['Survived'] == 0]
Data_s = Data[Data['Survived'] == 1]
features_means =list(Data.columns[0:10])
print(Data_d)
print(Data_s)
print(features_means)
print(Data.columns[0:10])


# In[9]:


outcome_count = Data.Survived.value_counts()
outcome_count = pd.Series(outcome_count)
outcome_count = pd.DataFrame(outcome_count)
outcome_count.index = ['alive', 'dead']

outcome_count['Percent'] = 100*outcome_count['Survived']/sum(outcome_count['Survived'])
outcome_count['Percent'] = outcome_count['Percent'].round().astype('int')
outcome_count


# In[10]:


sns.barplot(x = ['alive', 'dead'], y = 'Survived', data = outcome_count, alpha = .8)
plt.title('Frequency of Survived Outcomes in Dataset')
plt.ylabel('Frequency')
plt.show()


# In[11]:


sns.barplot(x = ['alive', 'dead'], y = 'Percent', data = outcome_count, alpha = .8)
plt.title('Percentage of Survived Outcomes in Dataset')
plt.ylabel('Percentage')
plt.ylim(0,100)
plt.show()


# In[12]:


fig = plt.figure(figsize=(10,20))
for i,b in enumerate(list(Data.columns[0:5])):
    
    i +=1
    ax = fig.add_subplot(3,4,i)
    
    sns.distplot(Data_s[b], kde=True, label='alive')
    sns.distplot(Data_d[b], kde=True, label='dead')
   
    ax.set_title(b)

sns.set_style("whitegrid")
plt.tight_layout()
plt.legend()
plt.show()   


# In[13]:


fig = plt.figure(figsize=(30,20))
for i,b in enumerate(list(Data.columns[0:5])):
    
    i +=1
    ax = fig.add_subplot(3,4,i)
    
    ax.hist(Data_s[b], label = 'alive', stacked = True, alpha=0.5, color= 'b')
    ax.hist(Data_d[b], label= 'dead', stacked = True, alpha=0.5, color= 'r')
    
    ax.set_title(b)

sns.set_style("whitegrid")
plt.tight_layout()
plt.legend()
plt.show()   


# In[14]:


fig = plt.figure(figsize=(20,10))
for i,b in enumerate(list(Data.columns[0:5])):
    i +=1
    
    ax = fig.add_subplot(3,4,i)
    ax.boxplot([Data_s[b], Data_d[b]])

    ax.set_title(b)

sns.set_style("whitegrid")
plt.tight_layout()
plt.legend()
plt.show()   


# In[15]:


sns.heatmap(Data.corr())
sns.set_style("whitegrid")
plt.show()


# In[16]:


X_test = pd.read_csv('test.csv')
X_test.head()


# In[17]:


X_test.drop(['PassengerId','Name','Embarked'], axis = 1, inplace = True)
X_test.drop(['Ticket','Fare','Cabin'], axis = 1, inplace = True)
Sex_coder = {'male':1, 'female':0}
X_test.Sex = X_test.Sex.map(Sex_coder)
X_test = X_test[pd.notnull(X_test['Age'])]
X_test.replace([np.inf, -np.inf], 0, inplace=True)
X_test.head()


# In[18]:


y_test = pd.read_csv('gender_submission.csv')
y_test.drop(['PassengerId'], axis = 1, inplace = True)
y_test.head()


# In[19]:


X_train=Data.iloc[:,:-1].values
y_train=Data['Survived'].values
norm = Normalizer()
norm.fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)


# In[20]:


SVM_params = {'C':[0.001, 0.1, 10, 100], 'kernel':['rbf' ,'linear', 'poly', 'sigmoid']}
LR_params = {'C':[0.001, 0.1, 1, 10, 100]}
LDA_params = {'n_components':[None, 1,2,3], 'solver':['svd'], 'shrinkage':[None]}
KNN_params = {'n_neighbors':[1,5,10,20, 50], 'p':[2], 'metric':['minkowski']}
RF_params = {'n_estimators':[10,50,100]}
DTC_params = {'criterion':['entropy', 'gini'], 'max_depth':[10, 50, 100]}


# In[21]:


models_opt = []

models_opt.append(('LR', LogisticRegression(), LR_params))
models_opt.append(('LDA', LinearDiscriminantAnalysis(), LDA_params))
models_opt.append(('KNN', KNeighborsClassifier(),KNN_params))
models_opt.append(('DTC', DecisionTreeClassifier(), DTC_params))
models_opt.append(('RFC', RandomForestClassifier(), RF_params))
models_opt.append(('SVM', SVC(), SVM_params))


# In[22]:


results = []
names = []


def estimator_function(parameter_dictionary, scoring = 'accuracy'):
    
    
    for name, model, params in models_opt:
    
        Kfold = KFold(len(X_train_norm), random_state=2, shuffle=True)

        model_grid = GridSearchCV(model, params)

        cv_results = cross_val_score(model_grid, X_train_norm, y_train, cv = Kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "Cross Validation Accuracy %s: Accarcy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())

        print(msg)


# In[23]:


estimator_function(models_opt, scoring = 'accuracy')


# In[24]:


GNB =  GaussianNB()
 
kfold = KFold(len(X_train_norm), random_state=2, shuffle=True)

cv_results_GNB= cross_val_score(GNB,X_train_norm, y_train, cv = kfold )

results.append(cv_results_GNB)
names.append('GNB')


# In[25]:


from sklearn.ensemble import VotingClassifier

estimators = []
model1 = LogisticRegression()

estimators.append(("logistic", model1))
model2 = DecisionTreeClassifier()
estimators.append(("cart", model2))
model3 = SVC()
estimators.append(("svm", model3))
model4 = KNeighborsClassifier()
estimators.append(("KNN", model4))
model5 = RandomForestClassifier()
estimators.append(("RFC", model5))
model6 = GaussianNB()
estimators.append(("GB", model6))
model7 = LinearDiscriminantAnalysis()
estimators.append(("LDA", model7))


voting = VotingClassifier(estimators)


results_voting = cross_val_score(voting, X_train_norm, y_train, cv=kfold)

results.append(results_voting)
names.append('Voting')

print('Accuracy: {} SD: {}'.format(results_voting.mean(), results_voting.std()))


# In[26]:


plt.boxplot(results, labels = names)
plt.title('Titanic survivals Accuracy using Various Machine Learning Models')
plt.ylabel('Model Accuracy %')
sns.set_style("whitegrid")
plt.ylim(1,1)
plt.show()


# In[27]:


lda_2 = LinearDiscriminantAnalysis()

lda_2.fit(X_train_norm, y_train)

lda_2_predicted = lda_2.predict(X_test_norm)


# In[28]:


print('Linear discriminant model analyis Accuracy is: {}'.format(accuracy_score(y_test,lda_2_predicted )))


# In[29]:


confusion_matrix_lda = pd.DataFrame(confusion_matrix(y_test, lda_2_predicted), index = ['Actual Negative','Actual Positive'], columns = ['Predicted Negative','Predicted Postive'] )

print('Linear discriminant Model Confusion Matrix')
confusion_matrix_lda


# In[30]:


print('Linear discriminant Model Classification Report')
print(classification_report(y_test, lda_2_predicted))


# In[31]:


RF_params = {'n_estimators':[10,50,100, 200]}

RFC_2 = RandomForestClassifier(random_state=42)

RFC_2_grid = GridSearchCV(RFC_2, RF_params)

RFC_2_grid.fit(X_train_norm, y_train)

print('Optimized number of estimators: {}'.format(RFC_2_grid.best_params_.values()))


# In[32]:


RFC_3 = RandomForestClassifier(n_estimators=50, random_state=42)

RFC_3.fit(X_train_norm, y_train)

RFC_3_predicted = RFC_3.predict(X_test_norm)
print('Model accuracy on test data: {}'.format(accuracy_score(y_test, RFC_3_predicted)))


# In[34]:


rfc_features = pd.DataFrame(list(zip(RFC_3.feature_importances_, Data.columns[:-1])), columns = ['Importance', 'Features'])

rfc_features = rfc_features.sort_values(['Importance'], ascending=False)


# In[35]:


sns.barplot(x = 'Importance', y = 'Features', data = rfc_features, )
plt.title('Feature Importance for Titanic survivals')
sns.set_style("whitegrid")
plt.show()


# In[36]:


lr_2 = LogisticRegression()
selector = RFECV(lr_2, cv = 10, scoring='accuracy')
selector.fit(X_train_norm, y_train)


# In[37]:


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Model Accuracy %")
plt.plot(selector.grid_scores_, alpha=0.8)
plt.tight_layout()
sns.set_style("whitegrid")

print('Logistic Regression Model Accuracy with Feature Elimination')
print('Optimal number of features: {}'.format(selector.n_features_))
print(len(selector.support_))
print(type(selector.support_))
selector.support_=np.append(selector.support_,True)
print((Data.columns))
print([i for i in list(Data.columns[selector.support_])])
plt.show()


# In[38]:


pca_var = PCA()

pca_var.fit(X_train_norm)

plt.plot(pca_var.explained_variance_, 'bo-', markersize=8)
plt.title("Elbow Curve for PCA Dimension of Titanic survivals Data")
plt.ylabel('Explained Variance')
plt.xlabel('Component Number')
sns.set_style("whitegrid")
plt.show()


# In[39]:


pca = PCA(n_components=3)

pca.fit(X_train_norm)
X_train_norm_pca = pca.transform(X_train_norm)

pca_df = pd.DataFrame(X_train_norm_pca, columns = ['PCA1', 'PCA2', 'PCA3'])

pca_df['Survived'] = y_train

pca_fig = plt.figure().gca(projection = '3d')
pca_fig.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c = pca_df['Survived'], cmap=cm.coolwarm)
pca_fig.set_xlabel('PCA1')
pca_fig.set_ylabel('PCA2')
pca_fig.set_zlabel('PCA3')
pca_fig.set_title('Data Visualized After 3-Component PCA')

sns.set_style("whitegrid")
plt.tight_layout()
plt.show()


# In[40]:



features = []

features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k='all')))

feature_union = FeatureUnion(features)

estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))

model_feature_union = Pipeline(estimators)

results_feature_union = cross_val_score(model_feature_union, X_train_norm, y_train, cv=kfold)


# In[41]:


results.append(results_feature_union)
names.append('LR-PCA')


# In[44]:


print('Mean accuracy is for logistic regression after PCA is: {} which is poorer than model accuracy without \ndimensional reduction.'.format(results_feature_union.mean()))


plt.boxplot( results, labels = names)
plt.ylabel('Model Accuracy %')
plt.ylim(1,1)
plt.title('Logistic Regression Performance after PCA Dimensional Reduction')
plt.show()


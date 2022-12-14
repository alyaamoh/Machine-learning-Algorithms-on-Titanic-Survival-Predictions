
# coding: utf-8

# In[1]:


## Import packages
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


# Load data from a csv file
Data = pd.read_csv('train.csv')
Data.head()


# In[3]:


# Drop unecessary columns
Data.drop(['PassengerId','Name','Embarked'], axis = 1, inplace = True)
Data.head(10)


# In[4]:


# Drop unecessary columns
Data.drop(['Ticket','Fare','Cabin'], axis = 1, inplace = True)
Data.head(10)


# In[5]:


# Numerize Survived "male" malignant; "female" benign using a dictionary and map function
Sex_coder = {'male':1, 'female':0}
Data.Sex = Data.Sex.map(Sex_coder)
Data.head()


# In[6]:


# Reorder columsn so diagnosis is right-most
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


# Create list of features related to the mean
Data_d = Data[Data['Survived'] == 0]
Data_s = Data[Data['Survived'] == 1]
features_means =list(Data.columns[0:10])
print(Data_d)
print(Data_s)
print(features_means)
print(Data.columns[0:10])


# In[9]:


#calculate how many survived or died
outcome_count = Data.Survived.value_counts()
outcome_count = pd.Series(outcome_count)
outcome_count = pd.DataFrame(outcome_count)
outcome_count.index = ['alive', 'dead']

outcome_count['Percent'] = 100*outcome_count['Survived']/sum(outcome_count['Survived'])
outcome_count['Percent'] = outcome_count['Percent'].round().astype('int')
outcome_count


# In[10]:


# Visualize frequency of alive/dead in dataset
sns.barplot(x = ['alive', 'dead'], y = 'Survived', data = outcome_count, alpha = .8)
plt.title('Frequency of Survived Outcomes in Dataset')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# Visualize Percentage of alive/dead in dataset
sns.barplot(x = ['alive', 'dead'], y = 'Percent', data = outcome_count, alpha = .8)
plt.title('Percentage of Survived Outcomes in Dataset')
plt.ylabel('Percentage')
plt.ylim(0,100)
plt.show()


# In[12]:


# Instantiate a figure object for OOP figure manipulation.
fig = plt.figure(figsize=(10,20))

# Create 'for loop' to enerate though Data features and compare with histograms
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


# Instantiate figure object
fig = plt.figure(figsize=(30,20))

# Create 'for loop' to enerate though Data features and compare with histograms
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

# Create 'for loop' to enerate though Data features and compare with histograms
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


# Quick visualization of relationships between features and diagnoses
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

# The normalize features to account for feature scaling
# Instantiate 
norm = Normalizer()
norm.fit(X_train)

# Transform both training and testing sets
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)


# In[20]:


# ## Model Testing
# * We'll start by testing a variety of algorithms using scikit-learns's gridsearch method for model optimization. 
# * Our models will include both parametric (e.g. SVM) and non-parametric (e.g. KNN) and linear (e.g. Logistic Regression) and nonlinear modles (e.g. Random Forest Classifier).

# Define parameters for optimization using dictionaries {parameter name: parameter list}
SVM_params = {'C':[0.001, 0.1, 10, 100], 'kernel':['rbf' ,'linear', 'poly', 'sigmoid']}
LR_params = {'C':[0.001, 0.1, 1, 10, 100]}
LDA_params = {'n_components':[None, 1,2,3], 'solver':['svd'], 'shrinkage':[None]}
KNN_params = {'n_neighbors':[1,5,10,20, 50], 'p':[2], 'metric':['minkowski']}
RF_params = {'n_estimators':[10,50,100]}
DTC_params = {'criterion':['entropy', 'gini'], 'max_depth':[10, 50, 100]}


# In[21]:


# Append list of models with parameter dictionaries
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


# Guassian Naive Bayes does not require optimization so we will run it separately without
# gridsearch and append the performance results to the results and names lists.

# Instantiate model
GNB =  GaussianNB()

# Define kfold - this was done above but not as a global variable
kfold = KFold(len(X_train_norm), random_state=2, shuffle=True)

# Run cross validation
cv_results_GNB= cross_val_score(GNB,X_train_norm, y_train, cv = kfold )

# Append results and names lists
results.append(cv_results_GNB)
names.append('GNB')


# In[25]:


# Ensemble Voting
from sklearn.ensemble import VotingClassifier

# Create list for estimatators
estimators = []
# Create estimator object
model1 = LogisticRegression()

# Append list with estimator name and object
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


# Visualize model accuracies for comparision - boxplots will be appropriate to visualize 
# data variation
plt.boxplot(results, labels = names)
plt.title('Titanic survivals Accuracy using Various Machine Learning Models')
plt.ylabel('Model Accuracy %')
sns.set_style("whitegrid")
plt.ylim(1,1)
plt.show()


# In[31]:


# ### Follow-up Model Testing
# * retry this model on test data after training on the entire training set.

# #### Test Random Forest Classifier
RF_params = {'n_estimators':[10,50,100, 200]}

RFC_2 = RandomForestClassifier(random_state=42)

# Instantiate gridsearch using RFC model and dictated parameters
RFC_2_grid = GridSearchCV(RFC_2, RF_params)

# Fit model to training data
RFC_2_grid.fit(X_train_norm, y_train)

print('Optimized number of estimators: {}'.format(RFC_2_grid.best_params_.values()))


# In[34]:


# Train RFC on whole training set

# Instantiate RFC with optimal parameters
RFC_3 = RandomForestClassifier(n_estimators=50, random_state=42)

RFC_3.fit(X_train_norm, y_train)

# Predict on training data using fitted RFC

# Evalaute RFC with test data
RFC_3_predicted = RFC_3.predict(X_test_norm)

# Create dataframe by zipping RFC feature importances and column names
rfc_features = pd.DataFrame(list(zip(RFC_3.feature_importances_, Data.columns[:-1])), columns = ['Importance', 'Features'])

# Sort in descending order for easy organization and visualization
rfc_features = rfc_features.sort_values(['Importance'], ascending=False)


# In[35]:


sns.barplot(x = 'Importance', y = 'Features', data = rfc_features, )
plt.title('Feature Importance for Titanic survivals')
sns.set_style("whitegrid")
plt.show()


# In[36]:


# ### Features Selection with Logistic Regression and Recursive Feature Elimination

# Instantiate new logistic regression for use with scikit-learn's recursive feature elimination...
# with cross validation (RFECV)
lr_2 = LogisticRegression()

# Instantiate RFECV with logistic regression classifier
selector = RFECV(lr_2, cv = 10, scoring='accuracy')

# Fit RFECV to training data
selector.fit(X_train_norm, y_train)


# In[37]:


#Plot number of features VS. cross-validation scores
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


# ## Dimensional Reduction

# Use dimensional reduction to reduce our 30 features into principal components (PCA) that maximally explain the data variance. While other forms of dimensional reduction exist (factor analysis, LDA, etc.), PCA is common approach and worth exploring with this data set.

# #### Evaluation of Variance with PCA Component Number

# Instantiate PCA
pca_var = PCA()

# Fit PCA to training data
pca_var.fit(X_train_norm)

# Visualize explained variance with an increasing number of components
plt.plot(pca_var.explained_variance_, 'bo-', markersize=8)
plt.title("Elbow Curve for PCA Dimension of Titanic survivals Data")
plt.ylabel('Explained Variance')
plt.xlabel('Component Number')
sns.set_style("whitegrid")
plt.show()


# In[39]:


# #### Visualization data by PCA  - 3D

#Instantiate new PCA object
pca = PCA(n_components=3)

# Fit and transform training data with PCA using 3 components
pca.fit(X_train_norm)
X_train_norm_pca = pca.transform(X_train_norm)

# Create a dataframe of 3 PCA
pca_df = pd.DataFrame(X_train_norm_pca, columns = ['PCA1', 'PCA2', 'PCA3'])

# Append diagnosis data into PCA dataframe
pca_df['Survived'] = y_train

# Visualize PCA in a 3D plot - color points by diagnsosis to see if a visuale stratification occurs
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


# Next, test logistic regression performance using PCA and scikit-learn's pipeline method.

# Create features list to use to instantiate the FeatureUnion
features = []

# Append features list
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k='all')))

# Instantiate FeatureUnion object
feature_union = FeatureUnion(features)

# Create pipeline using esimator list, append with feature union and logistic regression
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))

# Instantiate model using pipeline method
model_feature_union = Pipeline(estimators)

# Evaluate Pipeline model performance using cross validation
results_feature_union = cross_val_score(model_feature_union, X_train_norm, y_train, cv=kfold)


# In[41]:


# Append results and names lists
results.append(results_feature_union)
names.append('LR-PCA')


# In[44]:


print('Mean accuracy is for logistic regression after PCA is: {} which is poorer than model accuracy without \ndimensional reduction.'.format(results_feature_union.mean()))


plt.boxplot( results, labels = names)
plt.ylabel('Model Accuracy %')
plt.ylim(1,1)
plt.title('Logistic Regression Performance after PCA Dimensional Reduction')
plt.show()


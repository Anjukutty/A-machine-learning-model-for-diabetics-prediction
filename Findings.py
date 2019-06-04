
"""
Author: Anjukutty Joseph
Student ID: R00171244
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

def main():
    # read dataset
    pimaData = pd.read_csv("D:\Anjukutty\CIT\Semester2\MachineLearning\Assignment2\pima-indians-diabetes-database\diabetes.csv")
    print(pimaData.head())   
    expDataAnalysis(pimaData)    
    preProcessing2(pimaData)    
    pimaData= featureScalingNorm(pimaData) # Normalisation scaling
    #pimaData = featureScalingStand(pimaData) # standardisation scaling. Commented since normalisation is found to be better  
    train_set, test_set = createTrainAndTestSet(pimaData)
    trainModels(train_set)
    #bestModel = paraOptimiseSVC(train_set) # Attempted hyper parameter optimisation for SVC and LR. But LR is found to be better
    bestModel = paraOptimiseLR(train_set)
    predictDiabetics(bestModel, test_set)   # test on test data using optimised model.
    
    #training and testing best model on resampled data
    handleImbalanceSMOTE(pimaData,bestModel)    
    handleImbalanceADASYN (pimaData, bestModel)
    handleImbalanceRandom (pimaData, bestModel)
    
def expDataAnalysis(pimaData):    
    print("Shape of dataset:",pimaData.shape) # 768 rows, 9 columns
    # check distribution of target variable
    print("Number samples in both classes:",pimaData["Outcome"].value_counts()) # 0-500, 1-268 . imbalanced 
    sns.countplot(x='Outcome',data=pimaData)    
    plt.show()
    # check overall summary of the feature variables
    pd.set_option('display.max_columns', 100) # adjust display console to show full output
    print(pimaData.describe()) # some columns have 0 as minimum value which indicate missing data
    # Detailed examination of missing values in each column is performed
    missCol= [(((pimaData['Glucose']==0).value_counts())[1]),
              (((pimaData['BloodPressure']==0).value_counts())[1]),
              (((pimaData['SkinThickness']==0).value_counts())[1]),
              (((pimaData['Insulin']==0).value_counts())[1]),
              (((pimaData['BMI']==0).value_counts())[1])
              ]
    labels= ['Glucose','BP','SkinThickness','Insulin','BMI']    
    plt.bar(labels,missCol) # skin thickness and insulin have high number of missing values
    plt.show()
    
    # outlier detection  
    sns.boxplot(data = pimaData)
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show() 
    
def preProcessing2(pimaData):
    # create a dataframe for storing columns that has missing value analysis
    missingSubDf = pimaData[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','Outcome' ]]
    # check the mean of each column aganist outcome variable
    print("Mean of each column aganist outcome variable is shown below")
    print(missingSubDf.groupby('Outcome').mean())    
    
    # Replace missing value using mean for each class
    missingSubDf.loc[(missingSubDf['Outcome'] == 0 ) & (missingSubDf['Insulin']==0), 'Insulin'] =68.792
    missingSubDf.loc[(missingSubDf['Outcome'] == 1 ) & (missingSubDf['Insulin']==0), 'Insulin'] =100.335821
    missingSubDf.loc[(missingSubDf['Outcome'] == 0 ) & (missingSubDf['Glucose']==0), 'Insulin'] =109.980
    missingSubDf.loc[(missingSubDf['Outcome'] == 1 ) & (missingSubDf['Glucose']==0), 'Insulin'] =141.257463 
    missingSubDf.loc[(missingSubDf['Outcome'] == 0 ) & (missingSubDf['BloodPressure']==0), 'Insulin'] =68.184000 
    missingSubDf.loc[(missingSubDf['Outcome'] == 1 ) & (missingSubDf['BloodPressure']==0), 'Insulin'] =70.824627
    missingSubDf.loc[(missingSubDf['Outcome'] == 0 ) & (missingSubDf['SkinThickness']==0), 'Insulin'] =19.6640
    missingSubDf.loc[(missingSubDf['Outcome'] == 1 ) & (missingSubDf['SkinThickness']==0), 'Insulin'] =22.164179
    missingSubDf.loc[(missingSubDf['Outcome'] == 0 ) & (missingSubDf['BMI']==0), 'Insulin'] =30.304200
    missingSubDf.loc[(missingSubDf['Outcome'] == 1 ) & (missingSubDf['BMI']==0), 'Insulin'] =35.142537
    
    # Merge and return new dataframe
    temp = pd.DataFrame(pimaData[['Pregnancies','DiabetesPedigreeFunction','Age']])
    temp =pd.concat([temp,missingSubDf],sort=False,axis=1,join_axes =[missingSubDf.index])
    columnsTitles=["Pregnancies","Glucose","BloodPressure", "SkinThickness", "Insulin", "BMI","DiabetesPedigreeFunction","Age","Outcome"]
    temp=temp.reindex(columns=columnsTitles)    
    return temp
    
# This function apply feature scaling - normalisation technique    
def featureScalingNorm(pimaData):
    
    scalingObj= preprocessing.MinMaxScaler()
    X=  scalingObj.fit_transform(pimaData) # numpy array is output.
    temp = pd.DataFrame(X, columns = pimaData.columns)
    print(temp.head())
    return temp
  
# This function apply feature scaling - standardisation technique
def featureScalingStand(pimaData):   
     # remove label variable from standardisation.
     temp = pimaData.iloc[:,0:8]
     scalingObj = preprocessing.StandardScaler()
     X= scalingObj.fit_transform(temp)
     temp = pd.DataFrame(X, columns = temp.columns)     
     # concatenate outcome variable at last
     temp = pd.concat([temp, pimaData['Outcome']], axis =1, join_axes =[pimaData.index])
     print(temp.head())
     return temp
 
def createTrainAndTestSet(pimaData):
    #splitting using sklearn ,70% training data , 30% test data
    X_train,X_test,Y_train,Y_test = train_test_split(pimaData.iloc[:,0:8],pimaData['Outcome'],test_size = 0.3,
                                                 random_state = 41)
    train_set = pd.concat([X_train,Y_train],axis = 1,join_axes =[X_train.index])
    test_set = pd.concat([X_test,Y_test],axis = 1,join_axes =[X_test.index])
    train_set.to_csv("train_data.csv",sep = ',',encoding = 'Latin-1',index = False)
    test_set.to_csv("test_data.csv",sep = ',',encoding = 'Latin-1',index = False)
    return train_set,test_set

def trainModels(X_train):
    
    # create a dataframe to store details of each algorithms tried    
    data  = {'modelName':["LR","KNN","NB","SVC","RFC"],
             'modelObject':[LogisticRegression(),KNeighborsClassifier(),GaussianNB(),SVC(),RandomForestClassifier()],
             'accuracy':[None,None,None,None,None],
             'sd':[None,None,None,None,None]}   
    
    modelDetails =	pd.DataFrame(data,columns=['modelName',	'modelObject',	'accuracy','sd'])
    
    # Extract features from training data
    X_Features = X_train.iloc[:,0:8]
    # Extract labels from training data
    Y_Label = X_train['Outcome']
    # create and cross validate 5 models with algorithms mentioned above and store its accuracy    
    index =0
    for model in modelDetails['modelObject']:
        kfold = model_selection.KFold(n_splits=10, random_state=6)
        cv_results = model_selection.cross_val_score(model, X_Features, Y_Label, cv=kfold, scoring='accuracy')       
        modelDetails.iloc[index,2] = cv_results.mean() # take mean since kfold CV
        modelDetails.iloc[index,3] = cv_results.std()
        index= index+1
        
    ## print accuracy of models
    print("Accuracy of different models")
    print(modelDetails[['modelName','accuracy', 'sd']])
   
# This function does parameter optimisation for Logistic regression model
def paraOptimiseLR(X_train):
    # Extract features from training data
    X_Features = X_train.iloc[:,0:8]        
    # Extract labels from training data
    Y_Label = X_train['Outcome']
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  "penalty":["l1","l2"]}
    model_LR=LogisticRegression()    
    grid_search = GridSearchCV(model_LR, param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_Features, Y_Label)    
    LR= grid_search.best_estimator_ # choose the best estimator 
    print("grid search best params :",grid_search.best_params_)   
    #train model on best estimator
    LR.fit(X_Features, Y_Label)
    return LR
    
#Hyper parameter optimisation for SVC model. 
def paraOptimiseSVC(X_train):
    param_grid = {
    'C': [0.01,0.1,1.0, 10.0],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'shrinking': [True, False],
    'gamma': ['auto', 1, 0.1],
    'coef0': [0.0, 0.1, 0.5]
    }
    model_svc = SVC()
    # Extract features from training data
    X_Features = X_train.iloc[:,0:8]
    # Extract labels from training data
    Y_Label = X_train['Outcome']
    grid_search = GridSearchCV(
    model_svc, param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_Features, Y_Label)    
    svc = grid_search.best_estimator_ # choose the best estimator 
    print("grid search best score :",grid_search.best_score_)
    print("svc best params",grid_search.best_params_)
    #train model on best estimator
    svc.fit(X_Features, Y_Label)
    return svc

def predictDiabetics(model, test_set):
    Features = test_set.iloc[:,0:8]
    # Extract labels from training data
    Labels = test_set['Outcome']
    predicted = model.predict(Features)    
    print("F1 Score:", metrics.f1_score(Labels,predicted))
    print("Precision:", metrics.precision_score(Labels,predicted))
    print("Recall:", metrics.recall_score(Labels,predicted))
    print("Without resampling Confusion Matrix:", metrics.confusion_matrix(Labels,predicted))



def handleImbalanceSMOTE(pimaData,bestModel):
    train_Features = pimaData.iloc[:,0:8]
    # Extract labels from training data
    train_Labels = pimaData['Outcome']    
    sm = SMOTE(random_state=10)
    # Apply Resampling on entire dataset
    X_train, y_train= sm.fit_resample(train_Features, train_Labels)  
    # create training and test data
    X_train,X_test,Y_train,Y_test = train_test_split(X_train,y_train,test_size = 0.3,random_state = 12)    
    print("shape of model train", X_train.shape, "shape of test=", Y_train.shape)
    bestModel.fit(X_train, Y_train)
    predicted = bestModel.predict(X_test)    
    print("Model accuracy after resampling using SMOTE ")
    print("F1 Score:", metrics.f1_score(Y_test,predicted))
    print("Precision:", metrics.precision_score(Y_test,predicted))
    print("Recall:", metrics.recall_score(Y_test,predicted))
    print("Confusion Matrix:", metrics.confusion_matrix(Y_test,predicted))
    # Plot ROC Curve
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, predicted)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    

def handleImbalanceADASYN (pimaData, bestModel):
    train_Features = pimaData.iloc[:,0:8]
    # Extract labels from training data
    train_Labels = pimaData['Outcome']
    print('Original dataset shape %s' % Counter(train_Labels))
    ada  = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(train_Features, train_Labels)
    print('Resampled dataset shape %s' % Counter(y_res))    
    X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size = 0.3,random_state = 12)    
    bestModel.fit(X_train, Y_train)
    predicted = bestModel.predict(X_test)    
    print("Model accuracy after resampling using ADASYN")
    print("F1 Score:", metrics.f1_score(Y_test,predicted))
    print("Precision:", metrics.precision_score(Y_test,predicted))
    print("Recall:", metrics.recall_score(Y_test,predicted))
    
def handleImbalanceRandom (pimaData, bestModel):
    train_Features = pimaData.iloc[:,0:8]
    # Extract labels from training data
    train_Labels = pimaData['Outcome']
    print('Original dataset shape %s' % Counter(train_Labels))
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(train_Features, train_Labels)
    print('Resampled dataset shape %s' % Counter(y_res))    
    X_train,X_test,Y_train,Y_test = train_test_split(X_res,y_res,test_size = 0.3,random_state = 12)       
    bestModel.fit(X_train, Y_train)
    predicted = bestModel.predict(X_test)    
    print("Model accuracy after resampling using Random Over Sampling")
    print("F1 Score:", metrics.f1_score(Y_test,predicted))
    print("Precision:", metrics.precision_score(Y_test,predicted))
    print("Recall:", metrics.recall_score(Y_test,predicted))
    
main()
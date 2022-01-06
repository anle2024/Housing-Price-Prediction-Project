import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.linear_model import Lasso



import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")

    #demonstrateHelpers(trainDF)

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    #tuneGBR(trainInput, trainOutput, predictors)
    tuneLinearRidge(trainInput, trainOutput, predictors)
    #tuneLasso(trainInput, trainOutput, predictors)

    
    #doExperiment1(trainInput, trainOutput, predictors)
    
    #doExperiment2(trainInput, trainOutput, predictors, .15)
    
    #doExperiment3(trainInput, trainOutput, predictors, .15)
    #doExperiment3(trainInput, trainOutput, predictors, 5)
    #doExperiment3(trainInput, trainOutput, predictors, 10)
    
    
    #doExperiment4(trainInput, trainOutput, predictors, .15)
    #doExperiment4(trainInput, trainOutput, predictors, 5)
    #doExperiment4(trainInput, trainOutput, predictors, 10)


    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment1(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)

def doExperiment2(trainInput, trainOutput, predictors, x):
    gbrt=GradientBoostingRegressor(n_estimators=100,learning_rate=x) 
    cvMeanScore = model_selection.cross_val_score(gbrt, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    
    #print("GBR CV Average Score:", cvMeanScore)
    return cvMeanScore

def doExperiment3(trainInput, trainOutput, predictors, x):
    alg = linear_model.Ridge(alpha=x, copy_X=True, fit_intercept=True, max_iter=None, normalize=True, random_state=None, solver='auto', tol=0.001)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    # print("CV Average Score - Ridge " + str(x) + " :", cvMeanScore)
    return cvMeanScore

def doExperiment4(trainInput, trainOutput, predictors , x):
    alg = Lasso(alpha=x)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    #print("Lasso CV Average Score:" + str(x) + ":", cvMeanScore)
    return cvMeanScore
    


    
# ===============================================================================
'''
Tune parameters
'''
def tuneGBR(trainInput,trainOutput,predictors):
    
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeq = np.arange(.01,1,.01)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment2(trainInput, trainOutput, predictors,x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    plt.figure(figsize=(15,7))
    plt.plot(alphaList, acc)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Tunning of GBR')
    plt.grid(True)
    plt.savefig("TuningGBR.png", dpi=500, bbox_inches='tight')
    plt.show()
    
def tuneLinearRidge(trainInput,trainOutput,predictors):
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeq = np.arange(.01,1,.01)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment3(trainInput, trainOutput, predictors,x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    plt.figure(figsize=(15,7))
    plt.plot(alphaList, acc)
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')
    plt.title('Tunning of Linear Ridge')
    plt.grid(True)
    plt.savefig("TuningRidge.png", dpi=500, bbox_inches='tight')
    plt.show()
    
def tuneLasso(trainInput,trainOutput,predictors):
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeq = np.arange(.01,1,.01)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment4(trainInput, trainOutput, predictors, x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    plt.figure(figsize=(15,7))
    plt.plot(alphaList, ["{:.2f}".format(i) for i in acc])
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')
    plt.title('Tunning of Lasso')
    plt.grid(True)
    plt.savefig("TuningLasso.png", dpi=500, bbox_inches='tight')
    plt.show()

'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization
def dropColumn(df): #drop features that are considered useless
    
    #drop categorical and numerical columns that over 95% rows are missing  
    cat_col = df.select_dtypes(include=['object']).columns
    overfit_cat = []
    for i in cat_col:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 95:
            overfit_cat.append(i)

    overfit_cat = list(overfit_cat)
    df = df.drop(overfit_cat, axis=1)
    
    
    num_col = df.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns  #we will change type of MSSubClass later
    overfit_num = []
    for i in num_col:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 95:
            overfit_num.append(i)
    
    overfit_num = list(overfit_num)
    df = df.drop(overfit_num, axis =1)
    
    return df
    
    #2. 
    
#def droppingCol_list():    #to see which columns are already dropped so that we can avoid using them
#    trainDF = pd.read_csv("data/train.csv")
#    testDF = pd.read_csv("data/test.csv")
#    df_train = dropColumn(trainDF)
#    df_test = dropColumn(testDF)
#    print('dropping column in traindf: ', df_train.loc[0,:]) 
#    print('dropping column in traindf: ', df_test.loc[0,:])



def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)
    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
    return temp.loc[(temp['Total'] > 0)]

def HandleMissingValues(df):
    df['Functional'] = df['Functional'].fillna('Typ')
    df['Electrical'] = df['Electrical'].fillna('SBrkr') #Filling with modef
#  NA refers to "No Pool" (data description)
    df["PoolQC"] = df["PoolQC"].fillna("None")
# Replacing the missing values with 0, since no garage = no cars i
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['KitchenQual'] = df['KitchenQual'].fillna("TA")
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
# Replacing the missing values with None inferred from data dictionary 
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        df[col] = df[col].fillna('None')
# Replacing the missing values with None 
# NaN values for these categorical basement df_all, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')
#Replacing missing value it with median beacuse of outliers
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# Replacing the missing values with None 
# We have no particular intuition around how to fill in the rest of the categorical df_all
# So we replace their missing values with None
    objects = []
    for i in df.columns:
        if df[i].dtype == object:
            objects.append(i)
            df.update(df[objects].fillna('None'))

    numeric_dtypes = [ 'int64','float64']
    numerics = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numerics.append(i)
            df.update(df[numerics].fillna(0))

    df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    return df

def createDummy(df):
    
    df_num= df.select_dtypes(include=['float64','int64']).columns  # select numerical columns
    df_cat_temp = df.select_dtypes(exclude=['float64','int64']) # select object and categorical features 
    df_all_dummy= pd.get_dummies(df_cat_temp)
    df=pd.concat([df,df_all_dummy],axis=1) # joining converted dummy feature and original df_all dataset
    df= df.drop(df_cat_temp.columns,axis=1) #removing original categorical columns
    
    return df

    
    

def nonNumeric_to_String(df):    
    #the following column contains non numeric  
    df[['MSSubClass']] = df[['MSSubClass']].astype(str) 
    
'''
    featureEngineering code
'''

#def featureEngineer(df):
    
    # BEGIN: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    # EXPLANATION: using LabelEncoder to transform non-numerical ordinal values (i.e. 
    # Excellent, good, fair, etc).
#    lblEncodeList = ["KitchenQual", "GarageQual", "BsmtQual", "BsmtCond", "ExterQual", "GarageCond",
#                     "OverallCond", "FireplaceQu", "ExterCond", "HeatingQC", "BsmtFinType1", "BsmtFinType2"]
	# Label Encoder
#    for column in lblEncodeList:
#        lbl = LabelEncoder()
#        lbl.fit(list(df[column].values))
#        df[column] = lbl.transform(list(df[column].values))
    # END: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    
    
    #Create additional features that sum up other relevant features 
#    df['TotalBath'] = df['HalfBath'] + df['FullBath']    
#    df['TotalBsmSF'] = df['TotalBsmtSF'] + df['2ndFlrSF']
#    df['TotalPorch'] = df['OpenPorchSF']+df['EnclosedPorch'] +df['ScreenPorch'] 
#    df['TotalBsmtFin'] = df['BsmtFinSF1']+ df['BsmtFinSF2']
    
#    return df
    



def PerformOneHotEncoding(df,columnsToEncode):
    return pd.get_dummies(df,columns = columnsToEncode)





'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    dropColumn(trainDF)
    dropColumn(testDF)
    
    HandleMissingValues(trainDF)
    HandleMissingValues(testDF)
    
    nonNumeric_to_String(trainDF)
    nonNumeric_to_String(testDF)
    
    createDummy(trainDF)
    createDummy(testDF)
    
    
    #lblEncodeList = ["KitchenQual", "GarageQual", "BsmtQual", "BsmtCond", "ExterQual", "GarageCond",
    #                 "OverallCond", "FireplaceQu", "ExterCond", "HeatingQC", "BsmtFinType1", "BsmtFinType2"]
	# Label Encoder
    #for column in lblEncodeList:
    #    lbl = LabelEncoder()
    #    lbl.fit(list(trainDF[column].values))
    #    trainDF[column] = lbl.transform(list(trainDF[column].values))
    # END: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    
    
    #Create additional features that sum up other relevant features 
    TotalSF = ['TotalBsmtSF', '2ndFlrSF', '1stFlrSF', 'LotArea', 'GrLivArea','LowQualFinSF']
    TotalBath = ['FullBath', 'HalfBath']
    TotalPorch = ['OpenPorchSF', 'EnclosedPorch', 'ScreenPorch'] 
    TotalBsmtFin = ['BsmtFinSF1', 'BsmtFinSF2'] 
    qualitative_features = ['KitchenQual','BsmtQual','ExterQual','ExterCond','BsmtCond','HeatingQC']
    rooms = ['TotRmsAbvGrd','KitchenAbvGr','BedroomAbvGr']
    
    
    
    predictors = [ 'OverallQual', 'GrLivArea', 'GarageArea', 'YearBuilt', 'TotRmsAbvGrd'] + TotalSF + TotalBath + TotalPorch + TotalBsmtFin + qualitative_features + rooms
    
    

    
    
    #predictors = ['1stFlrSF', '2ndFlrSF']
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    
    for i in qualitative_features:
        ordinal_converting(trainInput, testInput, i)
    
   
    
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
# ===============================================================================
def ordinal_converting(trainDF,testDF,col):
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  4 if v=="Ex" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  3 if v=="Gd" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2 if v=="TA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1 if v=="Fa" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  0 if v=="Po" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="None" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: trainDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: trainDF.loc[:, col].mode().loc[0] if v=="None" else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])






def visualize_mostCorrelatedFeatures(): 
    trainDF = pd.read_csv("data/train.csv")
    corrmat = trainDF.corr()
    top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
    plt.figure(figsize=(10,10))
    sns.heatmap(trainDF[top_corr_features].corr(),annot=True,cmap="tab20")
    
    
    

'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()


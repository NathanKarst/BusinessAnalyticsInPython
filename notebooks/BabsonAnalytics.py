

def inspectLinearModel(train,trainTarget,model):
    import pandas as pd    
    import numpy as np
    from sklearn import feature_selection
    if type(model) == feature_selection.rfe.RFE:
        from sklearn import linear_model 
        train = train.loc[:,model.ranking_ == 1]
        model = linear_model.LinearRegression()
        model.fit(train,trainTarget)
    import numpy as np
    variables = list(train.columns)
    coefficients = list(model.coef_.round(2))
    if model.fit_intercept == True: 
        variables = ['(Intercept)'] + variables
        coefficients = [model.intercept_.round(2)] + coefficients
    
    p = slopePValues(train,trainTarget,model)
    
    df = pd.DataFrame({'Predictor':variables,'Estimate':coefficients,'p-value':p})
    df.set_index('Predictor',drop=True,inplace=True)
    
    print(df)
    print('\n\n')
    
    
def slopePValues(X,y,model):
    '''adapted from https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression'''
    import numpy as np
    from scipy import stats  
    import pandas as pd     

    X.reset_index(inplace=True,drop=True)
    
    X = X.select_dtypes(include=[np.number]) # only do p-values for numeric predictors right now


    params = np.append(model.intercept_,model.coef_)
    predictions = model.predict(X)
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
    
    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    
    return p_values


def benchmarkErrorRate(trainTarget,testTarget): 
    return sum(testTarget != trainTarget.value_counts().argmax())/len(testTarget)

def confusionMatrix(predictions, observations): 
    import pandas as pd
    return pd.crosstab(pd.Series(predictions).rename('Predictions'),observations.rename('Observations').reset_index(drop=True))


def makeDummies(X, forRegression=False):
    from itertools import chain
    import pandas as pd

    columns_ = X.columns
    cat_columns_ = X.select_dtypes(include=['category','object']).columns

    non_cat_columns_ = X.columns.drop(cat_columns_)

    startIdx = 0
    if forRegression: startIdx = 1

    cat_map_ = {col: X[col].cat.categories[startIdx:]
                     for col in cat_columns_}
    ordered_ = {col: X[col].cat.ordered
                     for col in cat_columns_}
    
    dummy_columns_ = {col: ["_".join([col, str(v)])
                                 for v in cat_map_[col]]
                           for col in cat_columns_}
    transformed_columns_ = pd.Index(
        non_cat_columns_.tolist() +
        list(chain.from_iterable(dummy_columns_[k]
                                 for k in cat_columns_))
    )
    
    return pd.get_dummies(X).reindex(columns=transformed_columns_).fillna(0)


def plotTree(model,train,trainTarget):
    from sklearn.tree import export_graphviz
    import graphviz

    model.fit(train,trainTarget)

    export_graphviz(model, out_file="mytree.dot",feature_names=train.columns,filled=True,rounded=True)
    with open("mytree.dot") as f:
        dot_graph = f.read()
    return graphviz.Source(dot_graph)


def crossValTree(model,train,trainTarget):
    from sklearn.model_selection import cross_val_score # version 0.18.1 and higher
    import matplotlib.pyplot as plt
    import numpy as np

    score_agg = []
    leaves = range(2,20)
    for leaf in leaves:
        model.max_leaf_nodes = leaf    
        scores = cross_val_score(model, train, trainTarget, cv=10)
        
        score_agg.append(np.mean(scores))
        
    plt.plot(leaves,score_agg,'o')
    plt.xlim((0,20))
    plt.xticks(leaves)
    plt.xlabel('Number of Leaves')
    plt.ylabel('Cross Validation Score')


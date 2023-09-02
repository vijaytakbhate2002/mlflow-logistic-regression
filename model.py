import logging
import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import sys
from urllib.parse import urlparse
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.base import ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

def confusionMatrix(true:np.ndarray,predicted:np.ndarray)->tuple:
    """ Args: true (numpy array), predicted (numpy array)
        Return: tuple (recall, precison, f1)"""
    recall = recall_score(true,predicted)
    precision = precision_score(true,predicted)
    f1 = f1_score(true,predicted)
    return recall, precision, f1

def ingestData(path:str)->pd.DataFrame:
    """ Reading path and returning dataframe """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error("Error during data fetching",e)
    return df

def crossValScore(algo:ClassifierMixin,X_train:np.ndarray,X_test:np.ndarray,y_train:np.ndarray,y_test:np.ndarray)->tuple:
    """ Args: X_train,X_test,y_train,y_test
        Return: Average testing and training score (train_score,test_score,cross_scores,predcited) """
    model = algo.fit(X_train,y_train)
    train_score = model.score(X_train,y_train)
    test_score = model.score(X_test,y_test)
    cross_scores = cross_val_score(algo,pd.concat([X_train,X_test],axis='rows'),pd.concat([y_train,y_test],axis='rows'))
    cross_scores = sum(cross_scores)/len(cross_scores)
    predicted = model.predict(X_test)
    return train_score,test_score,cross_scores,predicted,model

hyperparameters = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Regularization type
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'fit_intercept': [True, False],  # Whether to calculate the intercept for this model
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in the optimization problem
    'max_iter': [100, 200, 300, 400, 500],  # Maximum number of iterations for optimization
    'multi_class': ['auto', 'ovr', 'multinomial'],  # Strategy for handling multiple classes
    'warm_start': [True, False],  # Whether to reuse the solution of the previous call
    'random_state': [None, 42],  # Seed for random number generator
}

if __name__ == "__main__":
    df = ingestData("Datasets\Brain_stroke.csv")

    X = df.drop('target',axis='columns')
    y = df['target']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y)

    # Parameter accessing through system 
    penelty = str(sys.argv[1]) if len(sys.argv) > 1 else 'l2'
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    fit_intercept = eval(sys.argv[3]) if len(sys.argv) > 3 else True
    solver = str(sys.argv[4]) if len(sys.argv) > 4 else 'lbfgs'
    max_iter = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    multi_class = str(sys.argv[6]) if len(sys.argv) > 6 else 'auto'
    warm_start = eval(sys.argv[7]) if len(sys.argv) > 7 else False
    random_state = int(sys.argv[8]) if len(sys.argv) > 8 else None
    algo = LogisticRegression(penalty=penelty,C=C,fit_intercept=fit_intercept,solver=solver,max_iter=max_iter,multi_class=multi_class,warm_start=warm_start,random_state=random_state)

    (train_score,test_score,cross_scores,predicted,model) = crossValScore(algo,X_train,X_test,y_train,y_test)
    (recall, precision, f1) = confusionMatrix(y_test,predicted)


    with mlflow.start_run():
    # parameters
        mlflow.log_param('penelty',penelty)
        mlflow.log_param('C',C)
        mlflow.log_param('fit_intercept',fit_intercept)
        mlflow.log_param('solver',solver)
        mlflow.log_param('max_iter',max_iter)
        mlflow.log_param('multi_class',multi_class)
        mlflow.log_param('warm_start',warm_start)
        mlflow.log_param('random_state',random_state)
        # metrics
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("cross_scores", cross_scores)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)

        predictions = model.predict(X_train)
        signature = infer_signature(X_train,predictions)

        # # For remote server (DagsHub)
        # mlflow_uri = "https://dagshub.com/takbhatevijay/mlflow-tutorial.mlflow"
        # mlflow.set_tracking_uri(mlflow_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print("tracking_url_type_store",tracking_url_type_store)

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="LogisticRegression", signature=signature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)




import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os

def optimizar_hiperparametros(pipeline, x_train, y_train):
    param_grid = {
    "pca__n_components": [20, x_train.shape[1] - 2],
    'feature_selection__k': [12],
    'classifier__kernel': ["rbf"],
    'classifier__gamma': [0.1],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,         
        param_grid=param_grid,   
        cv=10,                       
        scoring='balanced_accuracy',  
                          
    )

    
    return grid_search

def save_model(path: str, estimator: GridSearchCV):
    with gzip.open(path, 'wb') as f:
        pickle.dump(estimator, f)

def pregunta01():
    # Cargar los datasets
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

    def cleanse(df):
        df = df.copy()
        df.rename(columns={'default payment next month': 'default'}, inplace=True)
        df.drop('ID', axis=1, inplace=True)
        df.dropna(inplace=True)
        df = df[(df["EDUCATION"]!=0) & (df["MARRIAGE"]!=0)]
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x>4 else x) 
        return df
    
    test_data=cleanse(test_data)
    train_data=cleanse(train_data)

    x_train=train_data.drop('default', axis=1)
    y_train=train_data['default']
    x_test=test_data.drop('default', axis=1)
    y_test=test_data['default']

    
    def f_pipeline(x_train):
        categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
        numerical_features = list(set(x_train.columns).difference(categorical_features))
        preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                    ("scaler", StandardScaler(with_mean=True, with_std=True), numerical_features),
                ],
                remainder='passthrough'
        )
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('pca', PCA()),
                ('feature_selection', SelectKBest(score_func=f_classif)),
                ('classifier', SVC(kernel="rbf", random_state=12345, max_iter=-1))
            ],
        )
        return pipeline
    pipeline = f_pipeline(x_train)

    grid_search = optimizar_hiperparametros(pipeline, x_train, y_train)
    grid_search.fit(x_train, y_train)

    path2 = "./files/models/"
    os.makedirs(path2, exist_ok=True)
    save_model(
        os.path.join(path2, 'model.pkl.gz'),
        grid_search,
    )
   

    pred_train = grid_search.predict(x_train)
    pred_test = grid_search.predict(x_test)

    def calc_metrics(y_true, y_pred, dataset):
        return {
            'type': 'metrics',
            'dataset': dataset,
            'precision': precision_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
    def matrix_calc(y_true, y_pred, dataset):
        cm = confusion_matrix(y_true, y_pred)
        return {
            'type': 'cm_matrix',
            'dataset': dataset,
            'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        }

    metrics = [
        calc_metrics(y_train, pred_train, 'train'),
        calc_metrics(y_test, pred_test, 'test'),
        matrix_calc(y_train, pred_train, 'train'),
        matrix_calc(y_test, pred_test, 'test')
    ]
    output_dir = "files/output"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
   

if __name__ == "__main__":
    pregunta01()
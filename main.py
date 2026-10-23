import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
def main():
    data = pd.read_csv(r"C:\Projekte\herzkrankheit-vorhersage\heart_disease_uci.csv")
    X = data.drop(columns=["num", "thal", "ca", "id", "dataset"])
    Y = data["num"]

    # Make Y binary
    Y = Y > 0
    Y = Y.astype(int)
    # Preprocession 
    # Missing Values    
    
    X["fbs"] = X["fbs"].fillna("Unknown")
    X["fbs"] = X["fbs"].astype("string")
    X["restecg"] = X["restecg"].fillna("Unknown")
    
    X["exang"] = X["exang"].fillna("Unknown")
    X["exang"] = X["exang"].astype("string")
    X["slope"] = X["slope"].fillna("Unknown")

    # A chol value of 0 can't be true
    X["chol"] = X["chol"].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state= 1, stratify=Y)

   
    
    




    # Pipeline 1. Fill Missing Values 2. Scale

    pl_numbers = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


    ct = ColumnTransformer([
        ("numbers", pl_numbers, ["age", "trestbps", "chol", "thalch", "oldpeak"]), 
        ("categories",  OneHotEncoder(handle_unknown="ignore"), ["restecg", "slope", "sex", "cp", "fbs", "exang"])]
    )


    lr_pipeline = make_pipeline(ct, LogisticRegression())
    rf_pipeline = make_pipeline(ct, RandomForestClassifier(random_state=1))
    xgb_pipeline = make_pipeline(ct, xgb.XGBClassifier())

    

    # Baseline and Logistic Regression
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    predict = dummy_clf.predict(X_test)
    print(f"Dummy Score: {dummy_clf.score(X_test, y_test):.3f}")

    # Display Confusion Matrix
    #display_confusion_matrix(lr_pipeline, X_test, y_test, ["Healthy", "Sick"], plt.cm.Blues)


    # It was important to create a baseline to see how much better the algorithm is than just guessing
    # Recall and Accuracy is a real problem you need to decide whether you want more people be worried or in real danger because of an false diagnose

    # Cross Validation
    
    models = {"Logistic Regression" : lr_pipeline, "Random Forest" : rf_pipeline, "XGBoost" : xgb_pipeline}
    scorings = ["accuracy", "recall"]
    for model in models:
        cross_validate(models[model], model, X_train, y_train, 5, scorings)
    

    # Validation Logistic Regression
    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)
    score = lr_pipeline.score(X_test, y_test)
    print(f"Logistic Regression Score: {score:.3f}")

    rc_score = recall_score(y_test, y_pred)
    print(f"Logistic Recall Score: {rc_score:.3f}")

    display_pipeline_weights(lr_pipeline)

    

def cross_validate(pipeline, name, x, y, cv, scorings):
    for scoring in scorings:
        cv_score = cross_val_score(pipeline, x, y, cv=cv, scoring=scoring)
        print(f"{name} {scoring} mean: {cv_score.mean():.3f}")
        print(f"{name} {scoring} standard deviation: {cv_score.std():.3f}")

def display_pipeline_weights(pipeline):
    weights = pd.DataFrame({
            "feature": [],
            "coef": [],
        })
    weights["feature"] = pipeline[0].get_feature_names_out()
    weights["coef"] = pipeline[-1].coef_[0]
    print(weights)
    

def display_confusion_matrix(model, x, y, labels, cmap):
    print("##### Wheights #####")
    disp = ConfusionMatrixDisplay.from_estimator(
            model,
            x,
            y,
            display_labels=labels,
            cmap=cmap,
            normalize="true",
        )
    disp.ax_.set_title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
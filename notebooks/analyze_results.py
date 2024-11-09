from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC
import joblib 
import pandas as pd 

X_train = pd.read_csv('../data/processed/X_train.csv')
X_test = pd.read_csv('../data/processed/X_test.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')

model = joblib.load('../models/XGBoost')

visualizer = ClassificationReport(model, classes=[0, 1], support=True)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 

visualizer = ConfusionMatrix(model, classes=[0, 1])

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 

visualizer = ROCAUC(model, classes=[0, 1])

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_train)

instance = 0

shap.plots.waterfall(shap_values[instance])

shap.plots.bar(shap_values)
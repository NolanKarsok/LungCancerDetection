**<h1>Lung Cancer Detection</h1>**

This repository contains code for binary classification of lung cancer using machine learning algorithms. The aim of this project is to develop a model that can predict whether a patient is likely to have lung cancer or not based on various clinical features. The code is written in Python and uses several popular machine learning libraries including Scikit-learn, Pandas, and NumPy.

**<h3>Dataset</h3>**
The dataset used in this project is obtained from Kaggle and generated from the online survey lung cancer system. The dataset is publicly available and contains information on patients diagnosed with lung cancer and consists of 15 features.

**<h3>Preprocessing</h3>**
Before training the machine learning model, the dataset is preprocessed using hot encoding, minority class resampling and feature selection.

**<h3>Model Selection</h3>**
Several machine learning algorithms are trained on the preprocessed dataset and evaluated using the accuracy, precision, recall, and F1-score metrics. The algorithms include Logistic Regression, K-Nearest Neighbor, Random Forest, Gradient Boosting, Support Vector Machine, Decision Tree. The best-performing algorithm is chosen for the final model.

**<h3>Performance</h3>**
The final model achieves an accuracy of 91%, precision of 91%, positive recall of 97%. The model is evaluated using k-fold cross-validation and the test set. The results suggest that the model is able to predict whether a patient is likely to have lung cancer or not with high accuracy and precision.

**<h3>Repository Structure</h3>**
* LungCancerDiagnosis.ipynb: contains complete Juypter Notebook and output for the project
* survey lung cancer.csv: raw data used in project
* Tooling.ipynb: Code for dashboard using Plotly that prompts user to respond to certain questions that represent the selected features in the final model. Using these features and final model selection, probaility of being labeled as positive or negative class is returned.
* Lung Cancer Binary Classification: Report summarizing all findings for project.

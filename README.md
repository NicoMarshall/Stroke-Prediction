# Stroke-Prediction
Using Support Vector Machines to classify patients according to whether they've suffered a stroke, based on their age, health data and lifestyle factors. Entirely coded in Python, making use of Pandas (data handling), Scikit-Learn (machine learning) and Matplotlib/Seaborn (visualization).

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Therefore, any model that can identify individuals at high risk would be highly useful in public health efforts and patient diagnostics for medical professionals. In this project we use Support Vector Machines (SVM) for this purpose, which fit a hyperplane to split the data into two regions; one for positive stroke and one for negative. SVMs are widely used in healthcare modelling due to their effectiveness on small datasets and binary classification problems such as this.

The three stages of this project are data cleaning, EDA (Exploratory Data Analysis) and model training; each has its own Jupyter notebook. The raw data is included in this repository ("healthcare-dataset-stroke-data.csv") as well as the cleaned version in a pickle file ("cleaned_data.pkl").

Data source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

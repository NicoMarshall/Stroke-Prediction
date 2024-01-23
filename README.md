# Stroke-Prediction
Using Support Vector Machines to classify patients according to whether they've suffered a stroke, based on their age, health data and lifestyle factors. Entirely coded in Python, making use of Pandas (data handling), Scikit-Learn (machine learning) and Matplotlib/Seaborn (visualization).

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Therefore, any model that can identify individuals at high risk would be highly useful in public health efforts and patient diagnostics for medical professionals. In this project we use Support Vector Machines (SVM) for this purpose, which fit a hyperplane to split the data into two regions; one for positive stroke and one for negative. SVMs are widely used in healthcare modelling due to their effectiveness on small datasets and binary classification problems such as this.

The three stages of this project are data cleaning, EDA (Exploratory Data Analysis) and model training; each has its own Jupyter notebook. The raw data is included in this repository ("healthcare-dataset-stroke-data.csv") as well as the cleaned version in a pickle file ("cleaned_data.pkl").

Data source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

# Stage 1: Data Cleaning
The raw data may have missing values, duplicates and outliers, which need to be either removed or augmented before a model can be trained.

Looking first at the numerical features, we choose to drop all missing values (since they amount to only 4% of records) and remove children from the data - they are at extremely low risk of stroke and might thus skew the data.

```

def remove_missing_numeric_values(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist() # select numeric columns
    mask = df["age"] > 17 # remove all children
    df = df[mask]
    df = df.dropna(subset=numeric_columns) # only bmi has missing values (4% of total). Drop these

    return df

```


Next, if we plot histograms of the continuous numerical features, we observe some unreasonable outliers for avg_glucose_level and bmi.
The likely explanation is mistakes in entering the data. 

![bmi_glucose_histograms](https://github.com/NicoMarshall/Stroke-Prediction/assets/109066030/99f8df3f-6e78-4df4-936c-fdb87015fc7e)


We will replace these outliers with the median for each feature:
```
def replace_outliers(df: pd.DataFrame):
    bmi_max = 50 #values above this will be replaced
    bmi_median = np.median(df["bmi"])
    avg_glucose_max = 180
    glucose_median = np.median(df["avg_glucose_level"])
    df["bmi"] = df["bmi"].apply(lambda x: bmi_median if x > bmi_max else x)
    df["avg_glucose_level"] = df["avg_glucose_level"].apply(lambda x: glucose_median if x > avg_glucose_max else x)

    return df
```
Finally, looking at the categorical features; only smoking status has missing values here, but these amount to 20% of total data. This is too high a proportion to simply drop, so we will just replace them with the mode for this column: "never smoked".

# Stage 2: EDA
We wish to visualise what relationships exist between our target variable ("stroke") and the features.

Experimenting with catplots of the numerical features, we see that the age of a patient has a clear difference in distribution, when we split the data based on the stroke variable:

![age_stroke_catplot](https://github.com/NicoMarshall/Stroke-Prediction/assets/109066030/922951d0-cea0-41d3-8184-7f5b0c38e9c4)

This can be verified statistically by doing t-tests for each feature to see if the means of the two populations (stroke vs no stroke) are  different, using a standard p-value threshold of 0.05:
```
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
numeric_columns.remove("id")
numeric_columns.remove("stroke")
pos_mask = (df["stroke"] > 0.5)
positive_stroke = df[pos_mask]
negative_stroke = df[~pos_mask]
for col in numeric_columns:
    t_statistic, p_value = ttest_ind(positive_stroke[col], negative_stroke[col], equal_var=False)
```
The results are that age, hypertension and heart disease meet the threshold for statistical significance. This is encouraging; three numerical features are strongly correlated with stroke, and may thus have good predicitve power when it comes to training classification models later.

Now doing similarly for our categorical features, we can split the data by category and stroke outcome (0 or 1). Any strong patterns will then hopefully become clear to see:
![non_stroke_categorical_pie_charts](https://github.com/NicoMarshall/Stroke-Prediction/assets/109066030/1ef59776-2f9e-4cd4-b67a-36d0fe65c324)


![stroke_categorical_pie_charts](https://github.com/NicoMarshall/Stroke-Prediction/assets/109066030/27784b71-7e87-48b1-914e-6605e1c81b37)



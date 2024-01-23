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



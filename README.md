# Stroke-Prediction
Using Support Vector Machines to classify patients according to whether they've suffered a stroke, based on their age, health data and lifestyle factors. 

All Jupyter Notebooks written in Python, making use of various data science libraries for data processing, machine learning and visualization;
  * Pandas
  * Numpy
  * Scikit-learn
  * Matplotlib
  * Seaborn

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


The differences in distribution are less pronounced here, with only ever_married and smoking_status exhibiting moderate significance. The marriage correlation could plausibly be simply as a weak proxy for age, which we already know to be correlated from the analysis above.

# Stage 3: Modelling

Since classical ML models like a SVMs require numerical inputs, we need to convert the categorical features (e.g gender) into numbers. For this we use Scikit-Learns in-built OneHotEncoder: 
```
def encode_categorical_features(dt: pd.DataFrame):
    """Use sklearns OneHotEncoder to encode categorical features

    Args:
        dt (pd.DataFrame): cleaned dataframe

    Returns:
        df_merged: dataframe of encoded categorical features as well as unchanged numerical ones
    """
    dt.drop("id", axis=1, inplace=True)
    text_columns = dt.select_dtypes(include=object).columns.tolist() #lists categorical features
    encoded_df = pd.DataFrame() # init empty dataframe to sequentially add encoded columns to
    for col in text_columns: #iterate through categorical features
        enc = OneHotEncoder(sparse_output=False) # init one hot encoder
        col_data = dt[[col]] # select column data
        enc_col = pd.DataFrame(enc.fit_transform(col_data), columns= enc.categories_) # fit encoder and transform data. Then convert to dataframe
        encoded_df = pd.concat([encoded_df, enc_col], axis=1) # add to encoded_df
       
           
    numeric_df= dt.select_dtypes(include=np.number) # select dataframe of numerical columns
    encoded_df.reset_index(inplace=True) # reset indices of both dataframes so they can be merged on the index column
    numeric_df.reset_index(inplace=True)
    df_merged = pd.merge(encoded_df, numeric_df, left_index=True, right_index=True) # merge numeric with encoded categorical
    df_merged.columns = df_merged.columns.astype(str) # cast column names to string, SVM doesn't like them otherwie
    df_merged = df_merged.drop("('index',)", axis = 1) # remove index columns 
    df_merged = df_merged.drop("index", axis = 1)

    return df_merged # return merged dataframe now ready for training
 
``` 
We can now split the data into a train and test set, and start training models with different hyperparameters. These can each be evaluated on the test set and compared.

A large issue here is the imbalanced classes in the dataset; positive stroke patients are only 5% of the total. Consequently, with the default hyperparameters the model simply learns to classify all patients as negative stroke and thus achieve a 95% accuracy; clearly this isn't our goal. One work around is to adjust the class_weight hyperparameter, which determines the regularization coefficients for the positive and negative classes. Effectively, this lets us tell the model to prioritise correctly classifying positive classes over negative ones; maximising the recall score at the expense of precision and accuracy. This is a reasonable choice because the cost of incorrectly classifying a positive case is far greater than incorrectly classifying a negative one. 

```
def train_svm(x_train, y_train):
    """Train SVM on train set

    Args:
        x_train 
        y_train 

    Returns:
        SVC_Gaussian: Trained Support vector machine, with rbf kernel and optimized class weight dict
    """
    SVC_Gaussian = SVC(kernel="rbf", class_weight={0:1, 1:10})
    SVC_Gaussian.fit(x_train, y_train)
    
    return SVC_Gaussian
```
With the above training setup, our model (saved as "model.joblib" in the rbf file) achieves the following metrics on the test set:
 * Accuracy: 83 %
 * Precision: 22 %
 * Recall: 56 %

In other words, just over half of stroke sufferers are identified by the model.

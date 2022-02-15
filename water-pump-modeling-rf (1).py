# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Data & Save to Delta Lake
# MAGIC ### - Specify schema
# MAGIC ### - Read from dbfs
# MAGIC ### - Write to 'bronze' delta lake

# COMMAND ----------

# Data source: https://www.kaggle.com/nphantawee/pump-sensor-data

# COMMAND ----------

dbfsPath = '/FileStore/tables/sensor_kaggle_dataset.csv'
deltaPathBronze = 'dbfs:/user/alexander.bonner@infinitive.com/predictive_platform/water_pump_test_bronze'

# COMMAND ----------

# specify schema of incoming data
schema = 'index INTEGER, timestamp TIMESTAMP, sensor_00 FLOAT, sensor_01 FLOAT, sensor_02 FLOAT, sensor_03 FLOAT, sensor_04 FLOAT, sensor_05 FLOAT, sensor_06 FLOAT, sensor_07 FLOAT, sensor_08 FLOAT, sensor_09 FLOAT, sensor_10 FLOAT, sensor_11 FLOAT, sensor_12 FLOAT, sensor_13 FLOAT, sensor_14 FLOAT, sensor_15 FLOAT, sensor_16 FLOAT, sensor_17 FLOAT, sensor_18 FLOAT, sensor_19 FLOAT, sensor_20 FLOAT, sensor_21 FLOAT, sensor_22 FLOAT, sensor_23 FLOAT, sensor_24 FLOAT, sensor_25 FLOAT, sensor_26 FLOAT, sensor_27 FLOAT, sensor_28 FLOAT, sensor_29 FLOAT, sensor_30 FLOAT, sensor_31 FLOAT, sensor_32 FLOAT, sensor_33 FLOAT, sensor_34 FLOAT, sensor_35 FLOAT, sensor_36 FLOAT, sensor_37 FLOAT, sensor_38 FLOAT, sensor_39 FLOAT, sensor_40 FLOAT, sensor_41 FLOAT, sensor_42 FLOAT, sensor_43 FLOAT, sensor_44 FLOAT, sensor_45 FLOAT, sensor_46 FLOAT, sensor_47 FLOAT, sensor_48 FLOAT, sensor_49 FLOAT, sensor_50 FLOAT, sensor_51 FLOAT, machine_status STRING'

# COMMAND ----------

# read in csv as sparkDF
sparkDF = (spark.read
           .format("csv")
           .option("header","true")
           .schema(schema)
           .option("sep", ",")
           .csv(dbfsPath)
          )

sparkDF = sparkDF.drop("index")

display(sparkDF)

# COMMAND ----------

# write df to delta lake
(sparkDF.write
 .format("delta")
 .mode("overwrite")
 .save(deltaPathBronze)
)

# COMMAND ----------

# verify read
sparkDF = (spark.read
           .format("delta")
           .load(deltaPathBronze)
          )

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Bronze Delta Path

# COMMAND ----------

deltaPathBronze = 'dbfs:/user/alexander.bonner@infinitive.com/predictive_platform/water_pump_test_bronze'

deltaPathSilver = 'dbfs:/user/alexander.bonner@infinitive.com/predictive_platform/water_pump_test_silver'

# COMMAND ----------

# read delta path as sparkDF
sparkDF = (spark.read
           .format("delta")
           .load(deltaPathBronze)
          )
display(sparkDF)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean Data
# MAGIC ### - remove columns with more than 50% nulls
# MAGIC ### - impute missing values with forward fill method
# MAGIC ### - Create a dataframe of records where 'machine_status' = 'Broken'

# COMMAND ----------

import pandas as pd

# convert spark df to pandas df
cleaned_sensor_df = sparkDF.toPandas()

# remove a dataframe column if more than a certain % of records are missing - set at > 50% initially
null_threshold = 50
threshold_count = int(((100-null_threshold)/100)*cleaned_sensor_df.shape[1] + 1)
cleaned_sensor_df = cleaned_sensor_df.dropna(axis=1, thresh=threshold_count)

# choose an imputation technique to replace null values - last or next populated value may make the most sense (method = 'ffill' or 'bfill'), but a more advanced method would be kmeans or other clustering method
cleaned_sensor_df.fillna(method='ffill', inplace=True) # replaces null with last populated value

# create new df of broken records
cleaned_broken_df = cleaned_sensor_df[cleaned_sensor_df['machine_status'] == 'BROKEN']

# COMMAND ----------

display(cleaned_sensor_df)
display(cleaned_broken_df)

# COMMAND ----------

# save cleaned df as silver delta lake
# save pandas df as spark df
cleanedDF = spark.createDataFrame(cleaned_sensor_df)

(cleanedDF.write
 .format("delta")
 .mode("overwrite")
 .save(deltaPathSilver)
)

# COMMAND ----------

# verify read of delta lake
silverSensorDF = (spark.read
                  .format("delta")
                  .load(deltaPathSilver)
                 )

display(silverSensorDF)

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Silver Delta Path

# COMMAND ----------

# read delta path as sparkDF
silverSensorDF = (spark.read
           .format("delta")
           .load(deltaPathSilver)
          )
display(silverSensorDF)

# COMMAND ----------

import pandas as pd
# create pandas df from spark df
silver_sensor_df = silverSensorDF.toPandas()

# create new df of broken records
silver_broken_df = silver_sensor_df[silver_sensor_df['machine_status'] == 'BROKEN']

# sort both DFs by 'timestamp' - was having issues plotting data below when the values were not sorted
silver_sensor_df.sort_values(by="timestamp", inplace=True)
silver_broken_df.sort_values(by="timestamp", inplace=True)

# COMMAND ----------

display(silver_sensor_df)
display(silver_broken_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Plot time series for each sensor with logged faults marked with X in red color
mpl.rcParams['agg.path.chunksize'] = 500000
sns.set_context('talk')



for column in silverSensorDF.columns:
    try:
        plt.figure(figsize=(18,3))
        plt.plot(silver_broken_df.timestamp, silver_broken_df[column], linestyle='none', marker='X', color='red', markersize=15)
        plt.plot(silver_sensor_df.timestamp, silver_sensor_df[column], color='blue')
        plt.title(column)
        plt.show()
    except:
        print("error with: {}".format(column))

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark_dist_explore import hist

# histogram - reads from spark df
for col in silverSensorDF.columns:
    try:
        print(f"========={col}=========")
        fig, ax = plt.subplots()
        hist(ax, silverSensorDF.select(col), bins = 50, color=['red'])
        display(fig)
    except:
        print("{}.....unable to plot".format(col))

# COMMAND ----------

# MAGIC %md
# MAGIC # Read Delta Path

# COMMAND ----------

deltaPathFE = 'dbfs:/user/alexander.bonner@infinitive.com/predictive_platform/water_pump_test_silver_FE'

# COMMAND ----------

# read delta path as sparkDF
silverSensorDF = (spark.read
           .format("delta")
           .load(deltaPathSilver)
          )

display(silverSensorDF)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering
# MAGIC ### - 'operational_hours' column derived from 'timestamp'
# MAGIC ### - 'machine_status' made binary for 'BROKEN' values only

# COMMAND ----------

import pandas as pd
import numpy as np 

# create pandas df from spark df
silver_sensor_df = silverSensorDF.toPandas()

# sort df by timestamp
silver_sensor_df.sort_values(by=['timestamp'], inplace=True)

# add 'operational_hours' column based on 'timestamp'
for i in range(silver_sensor_df.shape[0]):
    comparative_value = silver_sensor_df["timestamp"].values[i]
    base_value = silver_sensor_df["timestamp"].values[0]
    diff_value =  comparative_value - base_value
    diff_value = diff_value.astype('timedelta64[m]')
    diff_value = diff_value / np.timedelta64(1, 'm') / 60 # convert to hours
    silver_sensor_df.loc[silver_sensor_df.index[i], "operational_hr"] = diff_value
display(silver_sensor_df)

# COMMAND ----------

# create dummy columns for the 'machine_status' column - will only keep the BROKEN related one
silver_sensor_df = pd.get_dummies(silver_sensor_df, columns=["machine_status"])

# drop 'normal' and 'recovering' related columns
drop_col_list = ["machine_status_NORMAL", "machine_status_RECOVERING"]
silver_sensor_df.drop(columns=drop_col_list, inplace=True)

# convert to spark df
featureEngDF = spark.createDataFrame(silver_sensor_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Delta Lake - for feature engineering dataframe to be ingested by each modeling pipeline

# COMMAND ----------

(featureEngDF.write
.format("delta")
.mode("overwrite")
.save(deltaPathFE)
)

# COMMAND ----------

# verify delta read
df = (spark.read
 .format("delta")
 .load(deltaPathFE)
)

display(df)

# COMMAND ----------

# running notes of modeling methods tried

# SMOTE for creating synthetic records for the minority class
# Random undersampling for the majority class
# Bayesian optimization for model parameters

# COMMAND ----------

# MAGIC %md
# MAGIC # ML Pre-processing
# MAGIC ### - Drop 'timestamp' column
# MAGIC ### - Separate features and labels (classification model)
# MAGIC ### - Get features and labels into arrays
# MAGIC ### - Standardize and scale data

# COMMAND ----------

# drop unnecessary columns
drop_col_list = ["timestamp"]
silver_sensor_df.drop(columns=drop_col_list, inplace=True)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

label_col = ["machine_status_BROKEN"]

# get features df and array
df_features = silver_sensor_df.drop(label_col, axis=1)
features_array = np.array(df_features)

# get label df and array
df_label = silver_sensor_df[label_col]
label_array = np.array(df_label)

# scale features array
scaler = StandardScaler()
scaled_features_array = scaler.fit_transform(features_array)
print(scaled_features_array)

# COMMAND ----------

print(len(features_array))
print(len(label_array))

# COMMAND ----------

# incorporate random undersampling (for non-failures) and oversampling (for failures) for the imbalanced data set

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

sm = SMOTE(sampling_strategy=0.1, k_neighbors=3)
x_smote, y_smote = sm.fit_resample(scaled_features_array, label_array)

under = RandomUnderSampler(sampling_strategy=0.5)
x_over, y_over = under.fit_resample(x_smote, y_smote)

# COMMAND ----------

print(len(x_smote))
print(len(y_smote))
print(len(x_over))
print(len(y_over))

# COMMAND ----------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# principal component analysis - reduce number of features to X
pca = PCA(n_components=3) # n_components argument to be manually set after viewing plot of all PCA features
pcaFeatures = pca.fit_transform(x_over)

# plot principal components
features = range(pca.n_components_)
plt.figure(figsize=(15, 5))
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features)
plt.title("Importance of the Principal Components based on inertia")
plt.show()

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(pcaFeatures, y_over, test_size=.3, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling Pipeline
# MAGIC ### Model 1 - Random Forest Classifier

# COMMAND ----------

# Random Forest Classification Model Pipeline

from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from skopt import BayesSearchCV

# specify xgboost model and use grid search to fit to training set
rf  = RandomForestClassifier(random_state=42)

# set up ML Flow run with auto log-enable
mlflow.sklearn.autolog() 

with mlflow.start_run():
    
    # full parameter grid 
#     parameters = {
#         'n_estimators': [100, 120, 150],
#         'criterion': ['entropy', 'gini'],
#         'max_depth': [1,3,5,7,9],
#         'max_features': range(1,11),
#         'min_samples_split': range(2, 10),
#         'min_samples_leaf': [1,3,5]
#     }

    # limited parameter grid (to test MLFlow functionality)
#     parameters = {
#         'n_estimators': [50, 100, 200, 300],
#         'criterion': ['gini'],
#         'max_depth': [1,3,5],
#         'max_features': range(1,5),
#         'min_samples_split': range(2, 4),
#         'min_samples_leaf': [1,3,5]#,
#         'class_weight': ['balanced', '{0:0.0000317720, 1:0.9999682280}']
#     }
    
    # 3-fold cross-validator (grid search)
#     grid_search = GridSearchCV(
#         estimator=rf,
#         param_grid=parameters,
#         scoring = "f1",
#         n_jobs = -1,
#         cv = 3,
#         verbose=True)

    parameters = {
        'n_estimators': [50, 100, 200, 300],
        'criterion': ['gini'],
        'max_depth': [1,3],
        'max_features': [1,2,3], # can't exceed n_components from PCA
        'min_samples_split': [2,3,4]
    }

    
    # 3-fold cross-validator (bayesian optimization)
    grid_search = BayesSearchCV(
        estimator=rf,
        search_spaces=parameters,
        scoring = "f1",
        n_jobs = 10,
        cv = 3,
        verbose=True)

    cv_model = grid_search.fit(x_train, y_train.ravel())
    cv_pred_df = cv_model.predict(x_test)

    mlflow.sklearn.log_model(cv_model, "rf_best_model")

# COMMAND ----------

# Random Forest Classification Model Performance Metrics

import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn import model_selection

# visualize confusion matrix
matrix_labels = ['n_broken_status', 'y_broken_status']
conf_matrix = confusion_matrix(y_test, cv_pred_df)
print(conf_matrix)
plt.figure(figsize=(12,12))
sns.heatmap(conf_matrix, xticklabels=matrix_labels, yticklabels=matrix_labels, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()
# calculate the false positive rate from the confusion matrix
fp = conf_matrix[0][1]
tn = conf_matrix[1][1]
print("False Positive Rate: {}".format(fp / (fp+tn)))

# evaluate and summary performance metrics
print("The model used: {}".format("Random Forest Classifier"))
acc = accuracy_score(y_test, cv_pred_df)
print("Accuracy of model: {}".format(acc))
prec = precision_score(y_test, cv_pred_df, zero_division=1)
print("Precision: {}".format(prec))
recall = recall_score(y_test, cv_pred_df)
print("Recall: {}".format(recall))
f1 = f1_score(y_test, cv_pred_df)
print("F1 score: {}".format(f1))
mcc = matthews_corrcoef(y_test, cv_pred_df)
print("Matthews correlation coefficient: {}".format(mcc))

# COMMAND ----------

# use best model (based on most recent run model above) to make predicitons on the original data set

# must transform raw data to match principal components from the original data (without the over/undersampling included)
pca = PCA(n_components=3) # n_components argument to be manually set after viewing plot of all PCA features
pcaFeatures = pca.fit_transform(silver_sensor_df)

# make predictions and convert to pandas df
full_pred_df = grid_search.predict(pcaFeatures)
full_pred_df = pd.DataFrame(full_pred_df, columns=['broken_pred'])
fullpredDF = spark.createDataFrame(full_pred_df)
print(fullpredDF.count())
display(fullpredDF)

# COMMAND ----------

from pyspark.sql.functions import *
full_pred_df = pd.merge(silver_sensor_df, full_pred_df, how='left', left_index=True, right_index=True)
display(full_pred_df)

# COMMAND ----------

# compare observed failure to predicted failure 
test_df = full_pred_df.loc[(full_pred_df['machine_status_BROKEN']==1) | (full_pred_df['broken_pred']==1)]
test_df = full_pred_df.loc[(full_pred_df['broken_pred']==1)]
print(len(test_df))
display(test_df)

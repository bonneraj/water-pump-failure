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

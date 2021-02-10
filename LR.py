
# coding: utf-8

# In[2]:


import pandas as pd
from pyspark.ml.feature import StringIndexer,VectorAssembler, OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator , ParamGridBuilder


# In[3]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[6]:


spark = SparkSession.builder.getOrCreate()


# In[7]:


df_train = spark.read.csv("/bigdatads/common_folder/assignment3/car_data_train.csv",header=True)


# In[8]:


df_test = spark.read.csv("/bigdatads/common_folder/assignment3/car_data_test.csv",header=True)


# In[9]:


df_train.show(n=3)


# In[10]:


df_test.show(n=2)


# In[11]:


df_train.dtypes


# In[12]:


df_test.dtypes


# In[13]:


#casting the training numerical columns to Double
cast_cols = ['priceUSD','mileage(kilometers)','volume(cm3)']
for i in cast_cols:
    df_train = df_train.withColumn(i,df_train[i].cast("double"))


# In[14]:


#casting the testing numerical data columns to Double
for j in cast_cols:
    df_test = df_test.withColumn(j,df_test[j].cast("double"))


# In[15]:


df_train_new = df_train.drop('year','model','color')


# In[16]:


df_train_new.show(2)


# In[17]:


df_test_new = df_test.drop('year','model','color')


# In[18]:


df_test_new.show(2)


# In[19]:


categoricalColumns = [item[0] for item in df_train_new.dtypes if item[1].startswith('string')]


# In[20]:


categoricalColumns


# In[21]:


stages = []
#iterate through all categorical values
for categoricalCol in categoricalColumns:
    #create a string indexer for those categorical values and assign a new name including the word 'Index'
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    #append the string Indexer to our list of stages
    stages += [stringIndexer]


# In[22]:


#OneHot encoding the categorcial columns
OneHot_make = OneHotEncoder(inputCol='makeIndex',outputCol='make_feat')
OneHot_condition = OneHotEncoder(inputCol='conditionIndex',outputCol='condition_feat')
OneHot_Fueltype = OneHotEncoder(inputCol='fuel_typeIndex',outputCol='fuel_type_feat')
OneHot_transmission = OneHotEncoder(inputCol='transmissionIndex',outputCol='transmission_feat')
OneHot_driveunit =OneHotEncoder(inputCol='drive_unitIndex',outputCol='drive_unit_feat')


stages += [OneHot_make,OneHot_condition,OneHot_Fueltype,OneHot_transmission,OneHot_driveunit]


# In[23]:


stages


# In[24]:


numericalColumns = [item[0] for item in df_train_new.dtypes if item[1].startswith('double')]
numericalColumns


# In[25]:


assem_cols = ['makeIndex','conditionIndex','fuel_typeIndex','transmissionIndex','drive_unitIndex',
              'mileage(kilometers)','volume(cm3)','priceUSD']
assemble_features=VectorAssembler(inputCols=assem_cols,outputCol='features')


# In[26]:


stages += [assemble_features]


# In[27]:


#from pyspark.ml.feature import StandardScaler


# In[28]:


#target varibale
#stringIndexer_price = StringIndexer(inputCol='priceUSD',outputCol='label')
#scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")


# In[29]:


#scaler


# In[30]:


#stages +=[scaler]


# In[31]:


stages


# In[32]:


'''pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df_train_new)
train_model = pipelineModel.transform(df_train_new)'''


# In[33]:


#train_model.select('features').show(5)


# In[34]:


'''pipelineModel_test = pipeline.fit(df_test_new)
test_model = pipelineModel_test.transform(df_test_new)'''


# In[35]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


# In[36]:


#lr = LinearRegression(labelCol='priceUSD')


# In[37]:


lr = LinearRegression(labelCol='priceUSD')


# In[38]:


pipeline = Pipeline(stages=stages+[lr])


# In[39]:


stages+[lr]


# In[40]:


#Create the pipeline. Assign the satges list to the pipeline key word stages
#pipeline = Pipeline(stages = stages)


# In[41]:


lrevaluator = RegressionEvaluator(predictionCol="prediction", labelCol="priceUSD", metricName="rmse")


# In[42]:


paramGrid = ParamGridBuilder()    .addGrid(lr.maxIter,[10,15])    .addGrid(lr.regParam, [0.5, 0.05])     .build()


# In[43]:


crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=lrevaluator,
                          numFolds=3)


# In[44]:


cvModel = crossval.fit(df_train_new)


# In[45]:


df = cvModel.transform(df_train_new)


# In[46]:


df.select('priceUSD','features','prediction').show(5)


# In[47]:


pred = cvModel.transform(df_test_new)


# In[48]:


selected = pred.select('features','priceUSD','prediction')


# In[49]:


'''for row in selected.collect():
    print(row)'''


# In[50]:


cvModel.avgMetrics


# In[51]:


selected.show(5)


# In[52]:


rmse = lrevaluator.evaluate(pred)


# In[53]:


rmse = round(rmse,5)


# In[54]:


rmse


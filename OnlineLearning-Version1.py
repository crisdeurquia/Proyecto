#!/usr/bin/env python
# coding: utf-8

# In[87]:


#!/usr/bin/env python
# coding: utf-8
# Liberías

import findspark
findspark.init()

# Procesamiento de datos
import math
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import plotly.express as px
import json
from sklearn import preprocessing
from scipy.spatial import distance

# Reduccion dimensional

from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import umap


 # Deshabilitar alertas

import warnings
warnings.filterwarnings('ignore')

# Pyspark
import pyspark.sql.functions as F
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import ClusteringEvaluator


# Modelos de clustering 
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.feature import  StringIndexer, VectorAssembler, MinMaxScaler, OneHotEncoder, VectorSizeHint

#ELasticSearch 
from elasticsearch import Elasticsearch

from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel

#Base path
base_path='../../'

# Inicializa Sesion de PySpark
spark = SparkSession.builder         .appName('Prepro')         .config("spark.sql.session.timeZone", "Europe/London")         .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
sc = spark.sparkContext
print('> Inicializada la sesión de pyspark')


es = Elasticsearch(['http://138.4.7.132:9200/'])
index="bt"



def procesarDocumento(df):

    # suponiendo que hit['_source'] es un diccionario con las columnas 'version', 'timestamp', 'id', etc.
    row = Row(version=df['version'], 
              timestamp=df['timestamp'],
              id=df['id'],
              type=df['type'],
              event=df['event'],
              status=df['status'],
              classic_mode=df['classic_mode'],
              uuid=df['uuid'],
              company=df['company'],
              updated_at=df['updated_at'],
              last_seen=df['last_seen'],
              uap_lap=df['uap_lap'],
              address=df['address'],
              lmp_version=df['lmp_version'],
              le_mode=df['le_mode'],
              manufacturer=df['manufacturer'],
              created_at=df['created_at'],
              name=df['name'])

    # Crear el dataframe
    df = spark.createDataFrame([row])

    # Seleccionar las columnas necesarias
    df = df.select("version", "timestamp", "id", "type", "event", "status", "classic_mode", "uuid", "company", "updated_at", "last_seen", "uap_lap", "address", "lmp_version", "le_mode", "manufacturer", "created_at", "name")
    
    # Mostrar el dataframe
    df.show()
    
    df = primerPreprocesado(df)
    df = segundoPreprocesado(df)


# In[106]:


def primerPreprocesado(dataset):
    # Cambio de formato de las columnas temporales
    dataset = dataset.withColumn("timestamp", F.date_format("timestamp", "yyyy-MM-dd'T'HH:mm:ssXXX"))
    dataset= dataset.withColumn("last_seen", F.date_format("last_seen", "yyyy-MM-dd'T'HH:mm:ssXXX"))
    dataset = dataset.withColumn("created_at", F.date_format("created_at", "yyyy-MM-dd'T'HH:mm:ssXXX"))
    dataset = dataset.withColumn("updated_at", F.date_format("updated_at", "yyyy-MM-dd'T'HH:mm:ssXXX"))
    # Separamos en hora, minuto

    dataset = dataset.withColumn("hour_of_day", F.hour(F.col("last_seen")))
    dataset = dataset.withColumn("minute", F.minute(F.col("last_seen")))

    #dataset.agg(F.min('hour_of_day'), F.max('hour_of_day'), F.min('minute'), F.max('minute')).show()
    #dataset.groupby('hour_of_day').count().orderBy('hour_of_day').show(24)
    #dataset.orderBy('hour_of_day').show(vertical = True, truncate = False)

    print(f'Dimensiones del dataframe : {dataset.count()}, {len(dataset.columns)}')

        # Conversión sin,cos para mantener la distancia temporal

    dataset = dataset.withColumn("cos_time", F.cos(2 * np.pi * (dataset.hour_of_day + dataset.minute / 60.0) / 24))
    dataset = dataset.withColumn("sin_time", F.sin(2 * np.pi * (dataset.hour_of_day + dataset.minute / 60.0) / 24))

        # Transformación columnas status: online = 1 , offline = 0

    dataset = dataset.withColumn('status_index', F.when(F.col('status') == 'online', 1.0).otherwise(0.0))    

        # Transformación columnas classic_mode: t = 1 , f = 0

    dataset = dataset.withColumn('classic_mode_index', F.when(F.col('classic_mode') == 't', 1.0).otherwise(0.0))

        # Transformación columnas le_mode: t = 1 , f = 0

    dataset = dataset.withColumn('le_mode_index', F.when(F.col('le_mode') == 't', 1.0).otherwise(0.0))

        # Guardamos version de lmp_version

    dataset = dataset.withColumn('lmp_version_split', F.split(F.col('lmp_version'),"-").getItem(0))

    # Separamos nap, uap y lap de la columna address

    dataset = dataset.withColumn('nap_1', dataset.address.substr(1,2))
    dataset = dataset.withColumn('nap_1', F.expr('conv(nap_1, 16, 10)').cast(IntegerType()))

    dataset = dataset.withColumn('nap_2', dataset.address.substr(4,2))
    dataset = dataset.withColumn('nap_2', F.expr('conv(nap_2, 16, 10)').cast(IntegerType()))

    dataset = dataset.withColumn('uap', dataset.uap_lap.substr(1,2))
    dataset = dataset.withColumn('uap', F.expr('conv(uap, 16, 10)').cast(IntegerType()))

    dataset = dataset.withColumn('lap_1', dataset.uap_lap.substr(4,2))
    dataset = dataset.withColumn('lap_1', F.expr('conv(lap_1, 16, 10)').cast(IntegerType()))

    dataset = dataset.withColumn('lap_2', dataset.uap_lap.substr(7,2))
    dataset = dataset.withColumn('lap_2', F.expr('conv(lap_2, 16, 10)').cast(IntegerType()))

    dataset = dataset.withColumn('lap_3', dataset.uap_lap.substr(10,2))
    dataset = dataset.withColumn('lap_3', F.expr('conv(lap_3, 16, 10)').cast(IntegerType()))
    
    print('# Primer preprocesado realizado')
    return dataset


def segundoPreprocesado(dataset):
    columns_to_transform = ['nap_1','nap_2','uap','lap_1','lap_2','lap_3']
    
    # Escalado MinMax
    assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec", handleInvalid="skip") 
                  for col in columns_to_transform]
    scalers = [MinMaxScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in columns_to_transform]
    sizeHints = [VectorSizeHint(inputCol=col + "_scaled", handleInvalid="skip", size = 1) for col in columns_to_transform]

    pipeline = Pipeline(stages=assemblers + scalers + sizeHints)

    minMaxScaler_Model = pipeline.fit(dataset)
    dataset = minMaxScaler_Model.transform(dataset)

    minMaxScaler_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/minMaxScalerModel.bin"
    minMaxScaler_Model.write().overwrite().save(minMaxScaler_output_path)
    print('# MinMaxEscaler aplicado')
    
    # OneHotEncoding lmp_version
    stringIndexer = StringIndexer(inputCol = 'lmp_version_split', outputCol = 'lmp_version_index', handleInvalid="skip")
    ohe = OneHotEncoder(inputCol = 'lmp_version_index', outputCol = 'lmp_version_ohe', dropLast=False)

    pipeline = Pipeline(stages = [stringIndexer, ohe])

    Ohe_Model = pipeline.fit(dataset)
    dataset = Ohe_Model.transform(dataset)

    ohe_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/oheModel.bin"
    Ohe_Model.write().overwrite().save(ohe_output_path)
    print('# OneHotEncoding aplicado')

    # VectorAssembler
    vector_assembler = VectorAssembler(inputCols=['cos_time','sin_time','status_index','classic_mode_index','le_mode_index',
                                                  'lmp_version_ohe','nap_1_scaled','nap_2_scaled','uap_scaled','lap_1_scaled',
                                                  'lap_2_scaled','lap_3_scaled']
                                       ,outputCol = "features", handleInvalid="skip")

    dataset = vector_assembler.transform(dataset)

    vector_assembler_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/vectorAssemblerModel.bin"
    vector_assembler.write().overwrite().save(vector_assembler_output_path)
    print('# VectorAssembler aplicado')

    dataset.show(vertical=True)

    return dataset


def extraerDatos(es, index):
    query = {
        "query": {
            "match_all": {}
        }
    }
    response = es.search(
        index=index,
        scroll='2m',
        size=1,
        body=query
    )
    scroll_id = response['_scroll_id']  
    hit = response['hits']['hits'][0]
    
    # Procesar cada documento individualmente
    procesarDocumento(hit['_source'])
    
    #while len(response['hits']['hits']) > 0:
    #    for hit in response['hits']['hits']:
            # Procesar cada documento individualmente
           # procesarDocumento(hit['_source'])
       # response = es.scroll(scroll_id=scroll_id, scroll='2m')
    # scroll_id = response['_scroll_id']
        
extraerDatos(es,index)




#!/usr/bin/env python
# bin/elasticsearch --elastic
# 

# Librerias

import findspark
findspark.init()

    # Procesamiento de datos

import math
import random 
import numpy as np
from datetime import datetime
from scipy.spatial import distance

import pyspark.sql.functions as F
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

    # Modelos de clustering 

from pyspark.ml.clustering import KMeansModel


# Variables

APP_NAME = "SparkStreaming.py"
PERIOD = 10
BROKERS = '138.4.7.132:9092'
base_path = '.'
threshold = np.load(f'{base_path}/data/thresholdBt.npy',allow_pickle=True)
limit = 0

sc = SparkContext('local')
spark = SparkSession(sc)
spark.conf.set("spark.sql.session.timeZone", "Europe/Madrid")

# Lectura de datos de Elasticsearch
es_conf = {"es.nodes":"138.4.7.132", "es.port":"9200"}
    
def process_data(json_data):
    dataset = json.loads(json_data)
    dataset = spark.createDataFrame([dataset])
    
    
    dataset = dataset.withColumn("timestamp", F.from_unixtime(F.col("timestamp"), "yyyy-MM-dd'T'HH:mm:ssXXX"))
    dataset = dataset.withColumn("last_seen", F.from_unixtime(F.col("last_seen"), "yyyy-MM-dd'T'HH:mm:ssXXX"))
    dataset = dataset.withColumn("created_at", F.date_format(F.col("created_at"), "yyyy-MM-dd'T'HH:mm:ssXXX"))
    dataset = dataset.withColumn("updated_at", F.date_format(F.col("updated_at"), "yyyy-MM-dd'T'HH:mm:ssXXX"))

        # Creamos col hora_of_day, minute

    dataset = dataset.withColumn("hour_of_day", F.hour(F.col("last_seen")))
    dataset = dataset.withColumn("minute", F.minute(F.col("last_seen")))

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


    # OneHotEncoding lmp_version

    stringIndexer = StringIndexer(inputCol = 'lmp_version_split', outputCol = 'lmp_version_index', handleInvalid="skip")
    ohe = OneHotEncoder(inputCol = 'lmp_version_index', outputCol = 'lmp_version_ohe', dropLast=False)

    pipeline = Pipeline(stages = [stringIndexer, ohe])

    Ohe_Model = pipeline.fit(dataset)
    dataset = Ohe_Model.transform(dataset)

    ohe_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/oheModel.bin"
    Ohe_Model.write().overwrite().save(ohe_output_path)

    # VectorAssembler

    vector_assembler = VectorAssembler(inputCols=['cos_time','sin_time','status_index','classic_mode_index','le_mode_index',
                                                  'lmp_version_ohe','nap_1_scaled','nap_2_scaled','uap_scaled','lap_1_scaled',
                                                  'lap_2_scaled','lap_3_scaled']
                                       ,outputCol = "features", handleInvalid="skip")

    dataset = vector_assembler.transform(dataset)

    vector_assembler_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/vectorAssemblerModel.bin"
    vector_assembler.write().overwrite().save(vector_assembler_output_path)

    kmeans = KMeans(k=4, maxIter=100, tol=1e-4, distanceMeasure = 'euclidean', seed=319869)
    model = kmeans.fit(dataset.select('features'))

    lines = spark.readStream.format("kafka").option("kafka.bootstrap.servers", BROKERS).option("subscribe", PREDICTION_TOPIC).option("failOnDataLoss", "false").load()
    dataset =  lines.select(F.from_json(F.col("value").cast("string"),schema).alias("dataset")).select("dataset.*")

    # Función de actualización del modelo en línea
    def update_model_udf(features):
        global kmeans
        global centers
        model = kmeans
        centers = model.clusterCenters()
        prediction = model.predict(features)
        vectorCent = F.udf(lambda k: centroid(k,centers), ArrayType(DoubleType()))
        euclDistance = F.udf(lambda data,centroid: distToCentroid(data,centroid),FloatType())
        model = model.fit(features)
        return prediction, distance

    model_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/distanceKMeansBtModel.bin"
    model.write().overwrite().save(model_output_path)

vectorCent = F.udf(lambda k: centroid(k,centers), ArrayType(DoubleType()))
euclDistance = F.udf(lambda data,centroid: distToCentroid(data,centroid),FloatType())
 
while True:
    # Leer los datos entrantes de Elasticsearch
    raw_data = spark.read.format("org.elasticsearch.spark.sql").options(**es_conf).load("bt/doc-type")
    
    # Seleccionar las columnas necesarias y convertir los datos a formato JSON
    processed_data = raw_data.select("version",F.col("time").alias('timestamp'),"id","type","event" \
        ,"data.status","data.classic_mode","data.uuid","data.company","data.updated_at","data.last_seen","data.uap_lap","data.address","data.lmp_version" \
        ,"data.le_mode", "data.manufacturer", "data.created_at", "data.name").toJSON()

    # Procesar los datos y actualizar el modelo
    processed_data.foreach(process_data)

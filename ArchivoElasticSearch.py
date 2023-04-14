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
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import ClusteringEvaluator


# Modelos de clustering 
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.feature import  StringIndexer, VectorAssembler, MinMaxScaler, OneHotEncoder, VectorSizeHint

#ELasticSearch 
from elasticsearch import Elasticsearch

# Función de calculo de pertenencia a un cluster

def centroid (k,centers):
    return centers[k].tolist()

# Función de calculo de distancia euclidea al centroid

def distToCentroid(datapt, centroid):
    return distance.euclidean(datapt, centroid)

def plot_result (eval,param,nameX,nameY,title):
    plt.style.use('seaborn')
    
    fig,ax = plt.subplots()
    ax.plot(param,eval)
    
    ax.set_title(title)
    ax.set_xlabel(nameX)
    ax.set_ylabel(nameY)
    
    plt.show()
    
def round_double(val):
    return round(val, 10)

def anomalia (prediction, distance, threshold,limit):
    limit = min(limit,len(threshold[prediction]) - 1)
    if(distance > threshold[prediction][limit]):
        return True
    return False

def whitelistAnomalia (whitelist, id_entry, hour):
    detected = whitelist.detect_entry(id_entry,hour)
    if detected != None:
        return not detected
    return detected
    
# Definimos las funciones udf

to_array = F.udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

#Base path
base_path='../../'

# Inicializa Sesion de PySpark

spark = SparkSession.builder         .appName('Prepro')         .config("spark.sql.session.timeZone", "Europe/London")         .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
print('> Inicializada la sesión de pyspark')

# Whitelist
whitelist_path = '../../../whitelist/whitelist.json'
spark.sparkContext.addFile("../../../whitelist/module/whitelist.py")
# Cargar el modulo local
from whitelist import Whitelist

# Crear objeto Elasticsearch
es = Elasticsearch(['http://138.4.7.132:9200/'])

elastic_dataset = spark.read.format("org.elasticsearch.spark.sql")             .option("es.nodes", "138.4.7.132")             .option("es.port", "9200")             .option("es.resource", "bt/doc-type")             .load()
print('# Leidos datos de ElasticSearch')

dataset = elastic_dataset.select(
    F.col("version"),
    F.col("timestamp"),
    F.col("id"),
    F.col("type"),
    F.col("event"),
    F.struct(
        F.col("status"),
        F.col("classic_mode"),
        F.col("uuid"),
        F.col("company"),
        F.col("updated_at"),
        F.col("last_seen"),
        F.col("uap_lap"),
        F.col("address"),
        F.col("lmp_version"),
        F.col("le_mode"),
        F.col("manufacturer"),
        F.col("created_at"),
        F.col("name")
    ).alias("data")
)


dataset = dataset.select("version","timestamp","id","type","event"         ,"data.status","data.classic_mode","data.uuid","data.company","data.updated_at","data.last_seen","data.uap_lap","data.address","data.lmp_version"         ,"data.le_mode", "data.manufacturer", "data.created_at", "data.name")

print('# Dataset cargado')

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


# In[9]:


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


# In[10]:


# OneHotEncoding lmp_version

stringIndexer = StringIndexer(inputCol = 'lmp_version_split', outputCol = 'lmp_version_index', handleInvalid="skip")
ohe = OneHotEncoder(inputCol = 'lmp_version_index', outputCol = 'lmp_version_ohe', dropLast=False)

pipeline = Pipeline(stages = [stringIndexer, ohe])

Ohe_Model = pipeline.fit(dataset)
dataset = Ohe_Model.transform(dataset)

ohe_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/oheModel.bin"
Ohe_Model.write().overwrite().save(ohe_output_path)
print('# OneHotEncoding aplicado')


# In[11]:


# VectorAssembler

vector_assembler = VectorAssembler(inputCols=['cos_time','sin_time','status_index','classic_mode_index','le_mode_index',
                                              'lmp_version_ohe','nap_1_scaled','nap_2_scaled','uap_scaled','lap_1_scaled',
                                              'lap_2_scaled','lap_3_scaled']
                                   ,outputCol = "features", handleInvalid="skip")

dataset = vector_assembler.transform(dataset)

vector_assembler_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/vectorAssemblerModel.bin"
vector_assembler.write().overwrite().save(vector_assembler_output_path)

print('# VectorAssembler aplicado')


# In[12]:


dataset = dataset.withColumn('features', to_array('features'))
#def round_double(val):
#    return round(val, 10)
#round_udf = F.udf(lambda x: [round_double(val) for val in x], ArrayType(DoubleType()))
#dataset = dataset.withColumn('features', round_udf('features'))

#dataframe = dataset

#dataset = dataset.select('uuid','company','manufacturer','name' \
    #,'status','classic_mode','lmp_version','le_mode','created_at','updated_at','last_seen','uap_lap','address','features')
    
new_schema = ArrayType(DoubleType(), containsNull=False)
udf_foo = F.udf(lambda x:x, new_schema)
dataset = dataset.withColumn("features",udf_foo("features"))


# # Selección de modelos
# ## KMeans - Selección de hiperparámnetros
# #### Modelos a seleccionar:
# - KMeans
# - Bisecting KMeans
# - Gaussian Mixture Models

# In[13]:


feature_size = len(dataset.select('features').first()[0])


# In[14]:


# dataset = dataset.repartition(1) 

# KMeans: Seleccionando hiperparametro k

nClusters = [n for n in range(2,5,1)]
distanceMeasure = 'euclidean'
maxIter = 100 
tol=1e-4
seed=319869
silhouette = []
cost = []

for k in nClusters:
    kmeans = KMeans(k=k, maxIter=maxIter, tol=tol, distanceMeasure = distanceMeasure, seed=seed)
    model = kmeans.fit(dataset.select('features'))
    dataset_train = model.transform(dataset)
    cost.append(model.computeCost(dataset_train.select('features')))
    silhouette.append(ClusteringEvaluator().evaluate(dataset_train))
    print('-'*73)
    print(f'| k: {k:<3} | WSSE: {cost[-1]:<20} | Silhouette: {silhouette[-1]:<20}|')

plot_result(cost,nClusters,"nClusters",'Computer_Cost',"Variación del coste según nClusters")
plot_result(silhouette,nClusters,"nClusters",'Silhouette',"Variación de silhouette según nClusters")

# Punto de corte sse, silhouette

sse_scaled = preprocessing.minmax_scale(cost)
silh_scaled = preprocessing.minmax_scale(silhouette)

fig, ax = plt.subplots(figsize = (16,9))
ax.plot(nClusters, sse_scaled, c='blue', label = 'SSE escalado')
ax.plot(nClusters, silh_scaled, c= 'red', label = 'Silhouette escalado')

ax.axvline(x = 4, color='gray',linestyle='--', label = 'nCluster: 4')

ax.set_title('Comparacion escalada SSE vs silhouette')
ax.legend()

plt.show()

### Seleccionamos nClusters = 4



# In[49]:


# KMeans: Seleccionando hiperparametro distanceMeasure

nClusters = 4
distanceMeasure = ['euclidean','cosine']
maxIter = 100 
tol=1e-4
seed=319869
silhouette = []
cost = []

for dM in distanceMeasure:
    kmeans = KMeans(k=nClusters, maxIter=maxIter, tol=tol, distanceMeasure = dM, seed=seed)
    model = kmeans.fit(dataset.select('features'))
    dataset_train = model.transform(dataset)
    cost.append(model.computeCost(dataset_train.select('features')))
    silhouette.append(ClusteringEvaluator().evaluate(dataset_train))
    print('-'*73)
    print(f'| distanceMeasure: {dM:<15} | WSSE: {cost[-1]:<20} | Silhouette: {silhouette[-1]:<20}|')

plot_result(cost,distanceMeasure,"distanceMeasure",'Computer_Cost',"Variación del coste según distanceMeasure")
plot_result(silhouette,distanceMeasure,"distanceMeasure",'Silhouette',"Variación de silhouette según distanceMeasure")


# In[50]:


# KMeans: Seleccionando hiperparametro maxIter

nClusters = 4
distanceMeasure = 'euclidean'
maxIter = [n for n in range(100,1000,100)] 
tol=1e-4
seed=319869
silhouette = []
cost = []

for m in maxIter:
    kmeans = KMeans(k=nClusters, maxIter=m, tol=tol, distanceMeasure = distanceMeasure,  seed=seed)
    model = kmeans.fit(dataset.select('features'))
    dataset_train = model.transform(dataset)
    cost.append(model.computeCost(dataset_train.select('features')))
    silhouette.append(ClusteringEvaluator().evaluate(dataset_train))
    print('-'*73)
    print(f'| maxIter: {m:<5} | WSSE: {cost[-1]:<20} | Silhouette: {silhouette[-1]:<20}|')

plot_result(cost,maxIter,"maxIter",'Computer_Cost',"Variación del coste según maxIter")
plot_result(silhouette,maxIter,"maxIter",'Silhouette',"Variación de silhouette según maxIter")


# In[51]:


# KMeans: Seleccionando hiperparametro tol

nClusters = 4
distanceMeasure = 'euclidean'
maxIter = 100 
tol=[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
seed=319869
silhouette = []
cost = []

for t in tol:
    kmeans = KMeans(k=nClusters, maxIter=maxIter, tol=t, distanceMeasure = distanceMeasure, seed=seed)
    model = kmeans.fit(dataset.select('features'))
    dataset_train = model.transform(dataset)
    cost.append(model.computeCost(dataset_train.select('features')))
    silhouette.append(ClusteringEvaluator().evaluate(dataset_train))
    print('-'*80)
    print(f'| tol: {t:<7} | WSSE: {cost[-1]:<20} | Silhouette: {silhouette[-1]:<20}|')
    
plot_result(cost,tol,"tol",'Computer_Cost',"Variación del coste según tolerancia")
plot_result(silhouette,tol,"tol",'Silhouette',"Variación de silhouette según tolerancia")


# In[15]:


get_ipython().run_cell_magic('time', '', "\n# KMeans resultado\n# KMeans \n\nkmeans = KMeans(k=5, maxIter=100, tol=1e-4, distanceMeasure = 'euclidean', seed=319869)\nmodel_kmeans = kmeans.fit(dataset.select('features'))\n\ndataset_kmeans = model_kmeans.transform(dataset)\ncost_kmeans = model_kmeans.computeCost(dataset_kmeans.select('features'))\nsilhouette_kmeans = ClusteringEvaluator().evaluate(dataset_kmeans)\n\nprint(f'Tamaño del dataset: {dataset_kmeans.count(), len(dataset_kmeans.columns)}')")


# In[53]:


import umap

# Diferentes visualizaciones de los clusters: PCA, ISOMAP, TSNE Y UMAP.

vis_sample = 0.5
X = dataset_kmeans.sample(vis_sample, 123)

features = np.array(X.select('features').collect()).ravel().reshape(-1,feature_size)
Y = np.array(X.select('prediction').collect()).ravel().astype('float64')

pca_model = skPCA(n_components=2).fit_transform(features)
isomap_models = [Isomap(n_neighbors=n_nei, n_components=2, metric='euclidean', n_jobs = -1).fit_transform(features) 
                for n_nei in [10,20,35,50]]
tsne_models = [TSNE(n_components=2, perplexity=per, n_iter = 5000, metric='euclidean', n_jobs = -1).fit_transform(features) 
               for per in [10,20,35,50]]
umap_models = [umap.UMAP(n_components = n_nei, n_neighbors = n_nei, min_dist = min_dist, metric='euclidean', n_jobs = -1)
               .fit_transform(features) for n_nei in [10,20,35,50] for min_dist in [0.1,0.25,0.5,0.8]]


# In[54]:


# Plot de los diferentes reductores dimensionales

print(f'KMeans - clases: {np.unique(Y).size} [SSE:{cost_kmeans} | Silh: {silhouette_kmeans}]') 

plt.style.use('seaborn')

fig = plt.figure(figsize = (25,50))
fig.tight_layout(pad=2.0)

grid = plt.GridSpec(7, 4, hspace=0.2, wspace=0.15)

pca_plot = fig.add_subplot(grid[0, 1:3])
isomap_plots = [fig.add_subplot(grid[1, i:i+1]) for i in range(4)]
tsne_plots = [fig.add_subplot(grid[2, i:i+1]) for i in range(4)]
umap_plots = [fig.add_subplot(grid[i+3, j:j+1]) for i in range(4) for j in range(4)]

pca_plot.scatter(pca_model[:,0], pca_model[:,1], c = Y, cmap = 'rainbow')
pca_plot.set_title('PCA')
pca_plot.set_xlabel('feature_0')
pca_plot.set_ylabel('feature_1')

param_isomap = [n_nei for n_nei in [10,20,35,50]]

for index, model in enumerate(isomap_models):
    isomap_plots[index].scatter(model[:,0], model[:,1], c = Y, cmap = 'rainbow')
    isomap_plots[index].set_title(f'ISOMAP | n_nei:{param_isomap[index]}')
    isomap_plots[index].set_xlabel('feature_0')
    isomap_plots[index].set_ylabel('feature_1')
    
param_tsne = [per for per in [10,20,35,50]]
    
for index, model in enumerate(tsne_models):
    tsne_plots[index].scatter(model[:,0], model[:,1], c = Y, cmap = 'rainbow')
    tsne_plots[index].set_title(f'TSNE | perplexity:{param_tsne[index]}')
    tsne_plots[index].set_xlabel('feature_0')
    tsne_plots[index].set_ylabel('feature_1')
    
param_umap = [(n_nei,min_dist) for n_nei in [10,20,35,50] for min_dist in [0.1,0.25,0.5,0.8]]

for index, obj in enumerate(zip(umap_plots,umap_models)):
    obj[0].scatter(obj[1][:,0], obj[1][:,1], c = Y, cmap = 'rainbow')
    obj[0].set_title(f'UMAP | n_nei:{param_umap[index][0]}, min_dist:{param_umap[index][1]}')
    obj[0].set_xlabel('feature_0')
    obj[0].set_ylabel('feature_1')

plt.show()

# Mostramos parejas de características 

cols = ['cos_time','sin_time','status_index','classic_mode_index','le_mode_index','lmp_version_ohe','nap_1_scaled',
        'nap_2_scaled','uap_scaled','lap_1_scaled','lap_2_scaled','lap_3_scaled']

combinations = math.factorial(len(cols)) / (2 * math.factorial(len(cols) - 2))

ncols = 3
nrows = math.ceil(combinations / ncols) 

plt.style.use('seaborn')
fig = plt.figure(figsize= (21,7*nrows))
fig.tight_layout(pad=3.0)

j=1

for i, col_i in enumerate(cols):
    for col_j in cols[i+1:]:
        ax = fig.add_subplot(nrows, ncols, j)
        ax.scatter(features[:,cols.index(col_i)],features[:,cols.index(col_j)], cmap = 'rainbow' , c = Y)
        ax.set_title(f'{col_i} vs {col_j}')
        ax.set_xlabel(col_i)
        ax.set_ylabel(col_j)
        j+=1

        
dataset_kmeans.show(truncate = False, vertical = True)


# ## Bisecting KMeans - Selección de hiperparámetros

# In[16]:


# BKMeans: Seleccionando hiperparametro k

nClusters = [n for n in range(2,30,1)]
distanceMeasure = 'cosine'
maxIter = 100 
seed=319869
silhouette = []
cost = []

dataset = dataset.repartition(1) 


for k in nClusters:
    bkmeans = BisectingKMeans(k=k, maxIter=maxIter, distanceMeasure = distanceMeasure, seed=seed)
    model = bkmeans.fit(dataset.select('features'))
    dataset_bkmeans = model.transform(dataset)
    cost.append(model.computeCost(dataset_bkmeans.select('features')))
    silhouette.append(ClusteringEvaluator().evaluate(dataset_bkmeans))
    print('-'*73)
    print(f'| k: {k:<3} | WSSE: {cost[-1]:<20} | Silhouette: {silhouette[-1]:<20}|')
    
    
plot_result(cost,nClusters,"nClusters",'Computer_Cost',"Variación del coste según nClusters")
plot_result(silhouette,nClusters,"nClusters",'Silhouette',"Variación de silhouette según nClusters")

# Punto de corte sse, silhouette

sse_scaled = preprocessing.minmax_scale(cost)
silh_scaled = preprocessing.minmax_scale(silhouette)

fig, ax = plt.subplots(figsize = (16,9))

ax.plot(nClusters, sse_scaled, c='blue', label = 'SSE escalado')
ax.plot(nClusters, silh_scaled, c= 'red', label = 'Silhouette escalado')

ax.axvline(x = 2, color='gray',linestyle='--', label = 'nCluster: 2')

ax.set_title('Comparacion escalada SSE vs silhouette')
ax.legend(loc = 'upper center')

plt.show()


# In[17]:


# BKMeans: Seleccionando hiperparametro distanceMeasure

nClusters = 2
distanceMeasure = ['euclidean','cosine']
maxIter = 100 
seed=319869
silhouette = []
cost = []

for dM in distanceMeasure:
    bkmeans = BisectingKMeans(k=nClusters, maxIter=maxIter, distanceMeasure = dM, seed=seed)
    model = bkmeans.fit(dataset.select('features'))
    dataset_bkmeans = model.transform(dataset)
    cost.append(model.computeCost(dataset_bkmeans.select('features')))
    silhouette.append(ClusteringEvaluator().evaluate(dataset_bkmeans))
    print('-'*100)
    print(f'| distanceMeasure: {dM:<15} | WSSE: {cost[-1]:<20} | Silhouette: {silhouette[-1]:<20}|')

plot_result(cost,distanceMeasure,"distanceMeasure",'Computer_Cost',"Variación del coste según distanceMeasure")
plot_result(silhouette,distanceMeasure,"distanceMeasure",'Silhouette',"Variación de silhouette según distanceMeasure")


# In[18]:


# BKMeans: Seleccionando hiperparametro maxIter

nClusters = 2
distanceMeasure = 'cosine'
maxIter = [n for n in range(100,1000,100)] 
seed=319869
silhouette = []
cost = []

for m in maxIter:
    bkmeans = BisectingKMeans(k=nClusters, maxIter=m, distanceMeasure = distanceMeasure, seed=seed)
    model = bkmeans.fit(dataset.select('features'))
    dataset_bkmeans = model.transform(dataset)
    cost.append(model.computeCost(dataset_bkmeans.select('features')))
    silhouette.append(ClusteringEvaluator().evaluate(dataset_bkmeans))
    print('-'*80)
    print(f'| maxIter: {m:<5} | WSSE: {cost[-1]:<20} | Silhouette: {silhouette[-1]:<20}|')
    
plot_result(cost,maxIter,"maxIter",'Computer_Cost',"Variación del coste según maxIter")
plot_result(silhouette,maxIter,"maxIter",'Silhouette',"Variación de silhouette según maxIter")


# In[19]:


get_ipython().run_cell_magic('time', '', "# BKMeans \n\nbkmeans = BisectingKMeans(k = 2, maxIter=100, distanceMeasure = 'cosine', seed=319869)\nmodel_bkmeans = bkmeans.fit(dataset.select('features'))\ndataset_bkmeans = model_bkmeans.transform(dataset)\ncost_bkmeans = model_bkmeans.computeCost(dataset_bkmeans.select('features'))\nsilhouette_bkmeans = ClusteringEvaluator().evaluate(dataset_bkmeans)\n\n# Diferentes visualizaciones de los clusters: PCA, ISOMAP, TSNE Y UMAP.\n\nvis_sample = 1.0\nX = dataset_bkmeans.sample(vis_sample, 123)\n\nfeatures = np.array(X.select('features').collect()).ravel().reshape(-1,feature_size)\nY = np.array(X.select('prediction').collect()).ravel().astype('float64')\n\npca_model = skPCA(n_components=2).fit_transform(features)\nisomap_models = [Isomap(n_neighbors=n_nei, n_components=2, metric='cosine', n_jobs = -1).fit_transform(features) \n                for n_nei in [10,20,35,50]]\ntsne_models = [TSNE(n_components=2, perplexity=per, n_iter = 5000, metric='cosine', n_jobs = -1).fit_transform(features) \n               for per in [10,20,35,50]]\numap_models = [umap.UMAP(n_components = n_nei, n_neighbors = n_nei, min_dist = min_dist, metric='cosine', n_jobs = -1)\n               .fit_transform(features) for n_nei in [10,20,35,50] for min_dist in [0.1,0.25,0.5,0.8]]\n\n# Plot de los diferentes reductores dimensionales\n\nprint(f'BKMeans - clases: {np.unique(Y).size} [SSE:{cost_kmeans} | Silh: {silhouette_kmeans}]') \n\nplt.style.use('seaborn')\n\nfig = plt.figure(figsize = (25,50))\nfig.tight_layout(pad=2.0)\n\ngrid = plt.GridSpec(7, 4, hspace=0.2, wspace=0.15)\n\npca_plot = fig.add_subplot(grid[0, 1:3])\nisomap_plots = [fig.add_subplot(grid[1, i:i+1]) for i in range(4)]\ntsne_plots = [fig.add_subplot(grid[2, i:i+1]) for i in range(4)]\numap_plots = [fig.add_subplot(grid[i+3, j:j+1]) for i in range(4) for j in range(4)]\n\npca_plot.scatter(pca_model[:,0], pca_model[:,1], c = Y, cmap = 'rainbow')\npca_plot.set_title('PCA')\npca_plot.set_xlabel('feature_0')\npca_plot.set_ylabel('feature_1')\n\nparam_isomap = [n_nei for n_nei in [10,20,35,50]]\n\nfor index, model in enumerate(isomap_models):\n    isomap_plots[index].scatter(model[:,0], model[:,1], c = Y, cmap = 'rainbow')\n    isomap_plots[index].set_title(f'ISOMAP | n_nei:{param_isomap[index]}')\n    isomap_plots[index].set_xlabel('feature_0')\n    isomap_plots[index].set_ylabel('feature_1')\n    \nparam_tsne = [per for per in [10,20,35,50]]\n    \nfor index, model in enumerate(tsne_models):\n    tsne_plots[index].scatter(model[:,0], model[:,1], c = Y, cmap = 'rainbow')\n    tsne_plots[index].set_title(f'TSNE | perplexity:{param_tsne[index]}')\n    tsne_plots[index].set_xlabel('feature_0')\n    tsne_plots[index].set_ylabel('feature_1')\n    \nparam_umap = [(n_nei,min_dist) for n_nei in [10,20,35,50] for min_dist in [0.1,0.25,0.5,0.8]]\n\nfor index, obj in enumerate(zip(umap_plots,umap_models)):\n    obj[0].scatter(obj[1][:,0], obj[1][:,1], c = Y, cmap = 'rainbow')\n    obj[0].set_title(f'UMAP | n_nei:{param_umap[index][0]}, min_dist:{param_umap[index][1]}')\n    obj[0].set_xlabel('feature_0')\n    obj[0].set_ylabel('feature_1')\n\nplt.show()\n\n# Mostramos parejas de características \n\ncols = ['hour','mod_ohe','signal_scaled','freq_scaled','data_features']\n\ncombinations = math.factorial(len(cols)) / (2 * math.factorial(len(cols) - 2))\n\nncols = 3\nnrows = math.ceil(combinations / ncols) \n\nplt.style.use('seaborn')\nfig = plt.figure(figsize= (21,7*nrows))\nfig.tight_layout(pad=3.0)\n\nj=1\n\nfor i, col_i in enumerate(cols):\n    for col_j in cols[i+1:]:\n        ax = fig.add_subplot(nrows, ncols, j)\n        ax.scatter(features[:,cols.index(col_i)],features[:,cols.index(col_j)], cmap = 'rainbow' , c = Y)\n        ax.set_title(f'{col_i} vs {col_j}')\n        ax.set_xlabel(col_i)\n        ax.set_ylabel(col_j)\n        j+=1")


# ## Gaussian Mixture Model - Selección de hiperparámnetros

# In[20]:


# GMM: Seleccionando hiperparametro k

nClusters = [n for n in range(2,50,1)]
maxIter = 100 
tol=0.01
seed=319869

silhouette = []

for k in nClusters:
    gmm = GaussianMixture(k=k, maxIter=maxIter, tol=tol, seed=seed)
    model = gmm.fit(dataset.select('features'))
    dataset_gmm = model.transform(dataset)
    silhouette.append(ClusteringEvaluator().evaluate(dataset_gmm))
    print('-'*58)
    print(f'| k: {k:<3} | WSSE: None | Silhouette: {silhouette[-1]:<20}|')
    
plot_result(silhouette,nClusters,"nClusters",'Silhouette',"Variación de silhouette según nClusters")

# Punto de corte sse, silhouette

fig, ax = plt.subplots(figsize = (16,9))
ax.plot(nClusters, silhouette, c= 'red', label = 'Silhouette escalado')

ax.axvline(x = 2,color='gray',linestyle='--', label = 'nCluster: 2')

ax.set_title('Silhouette')
ax.legend()

plt.show()


# In[21]:


# GMM: Seleccionando hiperparametro maxIter

nClusters = 2
maxIter = [n for n in range(100,1000,100)] 
tol=0.01
seed=319869

silhouette = []

for m in maxIter:
    gmm = GaussianMixture(k=nClusters, maxIter=m, tol=tol, seed=seed)
    model = gmm.fit(dataset.select('features'))
    dataset_gmm = model.transform(dataset)
    silhouette.append(ClusteringEvaluator().evaluate(dataset_gmm))
    print('-'*58)
    print(f'| maxIter: {m:<3} | WSSE: None | Silhouette: {silhouette[-1]:<20}|')
    
plot_result(silhouette,maxIter,"maxIter",'Silhouette',"Variación de silhouette según maxIter")


# In[22]:


# GMM: Seleccionando hiperparametro tol

nClusters = 2
maxIter = 100 
tol=[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
seed=319869

silhouette = []

for t in tol:
    gmm = GaussianMixture(k=nClusters, maxIter=maxIter, tol=t, seed=seed)
    model = gmm.fit(dataset.select('features'))
    dataset_gmm = model.transform(dataset)
    silhouette.append(ClusteringEvaluator().evaluate(dataset_gmm))
    print('-'*65)
    print(f'| tol: {t:<8} | WSSE: None | Silhouette: {silhouette[-1]:<20}|')
    
plot_result(silhouette,tol,"tol",'Silhouette',"Variación de silhouette según tol")


# In[23]:


get_ipython().run_cell_magic('time', '', "# GMM\n\ngmm = GaussianMixture(k=2, seed=319869)\nmodel_gmm = gmm.fit(dataset.select('features'))\ndataset_gmm = model_gmm.transform(dataset)\nsilhouette_gmm = ClusteringEvaluator().evaluate(dataset_gmm)\n\n# Diferentes visualizaciones de los clusters: PCA, ISOMAP, TSNE Y UMAP.\n\nvis_sample = 1.0\nX = dataset_gmm.sample(vis_sample, 123)\n\nfeatures = np.array(X.select('features').collect()).ravel().reshape(-1,feature_size)\nY = np.array(X.select('prediction').collect()).ravel().astype('float64')\n\npca_model = skPCA(n_components=2).fit_transform(features)\nisomap_models = [Isomap(n_neighbors=n_nei, n_components=2, metric='cosine', n_jobs = -1).fit_transform(features) \n                for n_nei in [10,20,35,50]]\ntsne_models = [TSNE(n_components=2, perplexity=per, n_iter = 5000, metric='cosine', n_jobs = -1).fit_transform(features) \n               for per in [10,20,35,50]]\numap_models = [umap.UMAP(n_components = n_nei, n_neighbors = n_nei, min_dist = min_dist, metric='cosine', n_jobs = -1)\n               .fit_transform(features) for n_nei in [10,20,35,50] for min_dist in [0.1,0.25,0.5,0.8]]\n\n# Plot de los diferentes reductores dimensionales\n\nprint(f'GMM - clases: {np.unique(Y).size} [SSE:{None} | Silh: {silhouette_gmm}]') \n\nplt.style.use('seaborn')\n\nfig = plt.figure(figsize = (25,50))\nfig.tight_layout(pad=2.0)\n\ngrid = plt.GridSpec(7, 4, hspace=0.2, wspace=0.15)\n\npca_plot = fig.add_subplot(grid[0, 1:3])\nisomap_plots = [fig.add_subplot(grid[1, i:i+1]) for i in range(4)]\ntsne_plots = [fig.add_subplot(grid[2, i:i+1]) for i in range(4)]\numap_plots = [fig.add_subplot(grid[i+3, j:j+1]) for i in range(4) for j in range(4)]\n\npca_plot.scatter(pca_model[:,0], pca_model[:,1], c = Y, cmap = 'rainbow')\npca_plot.set_title('PCA')\npca_plot.set_xlabel('feature_0')\npca_plot.set_ylabel('feature_1')\n\nparam_isomap = [n_nei for n_nei in [10,20,35,50]]\n\nfor index, model in enumerate(isomap_models):\n    isomap_plots[index].scatter(model[:,0], model[:,1], c = Y, cmap = 'rainbow')\n    isomap_plots[index].set_title(f'ISOMAP | n_nei:{param_isomap[index]}')\n    isomap_plots[index].set_xlabel('feature_0')\n    isomap_plots[index].set_ylabel('feature_1')\n    \nparam_tsne = [per for per in [10,20,35,50]]\n    \nfor index, model in enumerate(tsne_models):\n    tsne_plots[index].scatter(model[:,0], model[:,1], c = Y, cmap = 'rainbow')\n    tsne_plots[index].set_title(f'TSNE | perplexity:{param_tsne[index]}')\n    tsne_plots[index].set_xlabel('feature_0')\n    tsne_plots[index].set_ylabel('feature_1')\n    \nparam_umap = [(n_nei,min_dist) for n_nei in [10,20,35,50] for min_dist in [0.1,0.25,0.5,0.8]]\n\nfor index, obj in enumerate(zip(umap_plots,umap_models)):\n    obj[0].scatter(obj[1][:,0], obj[1][:,1], c = Y, cmap = 'rainbow')\n    obj[0].set_title(f'UMAP | n_nei:{param_umap[index][0]}, min_dist:{param_umap[index][1]}')\n    obj[0].set_xlabel('feature_0')\n    obj[0].set_ylabel('feature_1')\n\nplt.show()\n\n\n# Mostramos parejas de características \n\ncols = ['hour','mod_ohe','signal_scaled','freq_scaled','data_features']\n\ncombinations = math.factorial(len(cols)) / (2 * math.factorial(len(cols) - 2))\n\nncols = 3\nnrows = math.ceil(combinations / ncols) \n\nplt.style.use('seaborn')\nfig = plt.figure(figsize= (21,7*nrows))\nfig.tight_layout(pad=3.0)\n\nj=1\n\nfor i, col_i in enumerate(cols):\n    for col_j in cols[i+1:]:\n        ax = fig.add_subplot(nrows, ncols, j)\n        ax.scatter(features[:,cols.index(col_i)],features[:,cols.index(col_j)], cmap = 'rainbow' , c = Y)\n        ax.set_title(f'{col_i} vs {col_j}')\n        ax.set_xlabel(col_i)\n        ax.set_ylabel(col_j)\n        j+=1\n        \ndataset_gmm.show()")


# ## Comparación de los 3 modelos

# In[24]:


# Metricas

print(f'Resultado KMeans óptimo: K = {model_kmeans.summary.k}, SSE = {cost_kmeans}, Silhoutte  = {silhouette_kmeans}')
print(f'Resultado BKMeans óptimo: K = {model_bkmeans.summary.k}, SSE = {cost_bkmeans}, Silhoutte  = {silhouette_bkmeans}')
print(f'Resultado GMM óptimo: K = {model_gmm.summary.k}, SSE = {None} , Silhoutte  = {silhouette_gmm}')


# ## Modelo seleccionado: KMeans

# # Entrenamiento

# In[25]:


# Entrenamiento y predicción

kmeans = KMeans(k=4, maxIter=100, tol=1e-4, distanceMeasure = 'euclidean', seed=319869)
model = kmeans.fit(dataset.select('features'))
model_output_path = f"{base_path}StructuredStreaming/DistanceKMeans/data/distanceKMeansBtModel.bin"
model.write().overwrite().save(model_output_path)

train = model.transform(dataset.select('features'))

# Evaluación de resultados

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(train)
computeCost = model.computeCost(train.select('features'))
print(f'Silhouette : {silhouette}')
print(f'SSE        : {computeCost} \n')

centers = model.clusterCenters()
vectorCent = F.udf(lambda k: centroid(k,centers), ArrayType(DoubleType()))
euclDistance = F.udf(lambda data,centroid: distToCentroid(data,centroid),FloatType())


# In[26]:


# Calculamos valor del centroid más cercano.

train = train.withColumn('centroid', vectorCent(F.col('prediction')))

# Calculamos distancia al centroid más cercano.

train = train.withColumn('distance', euclDistance(F.col('features'),F.col('centroid')))

threshold = train.groupBy('prediction').agg(F.sort_array(F.collect_list('distance'), asc=False).alias('distances')).orderBy('prediction')

threshold.show()


# In[27]:


threshold_values = threshold.select('distances').toPandas()['distances'].values
threshold_path = f'data/thresholdBt.npy'
np.save(threshold_path, threshold_values)


# # Test

# In[ ]:


# new_schema = ArrayType(DoubleType(), containsNull=False)
# udf_foo = F.udf(lambda x:x, new_schema)

# dataframe = dataframe.withColumn("features",udf_foo("features"))


# In[ ]:


#predictions = model.transform(dataframe)
test = model.transform(dataframe)
test = model.transform(dataset)

# Evaluación de resultados

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(test)
computeCost = model.computeCost(test.select('features'))
print(f'Silhouette : {silhouette}')
print(f'SSE        : {computeCost} \n')

centers = model.clusterCenters()

# Calculamos valor del centroid más cercano.

test = test.withColumn('centroid', vectorCent(F.col('prediction')))

# Calculamos distancia al centroid más cercano.

test = test.withColumn('distance', euclDistance(F.col('features'),F.col('centroid')))

train.groupBy('prediction').max('distance').orderBy('prediction').show(truncate=False)

test.groupBy('prediction').min('distance').orderBy('prediction').show(truncate=False)

limit = 0


detectAnom = F.udf(lambda prediction, distance: anomalia(prediction, distance, threshold_values, limit), BooleanType())


# Creamos columna ANOMALIA, teniendo valores booleanos

test = test.withColumn('anomalia_model', detectAnom(F.col('prediction'),F.col('distance')))
test.select('prediction','distance','anomalia_model').show(100,truncate=False)


# ## Whitelist

print('> Cargando whitelist')

whitelist = Whitelist(whitelist_path)
whitelistAnom = F.udf(lambda i,h: whitelistAnomalia(whitelist,i,h), BooleanType())
test = test.withColumn('anomalia_whitelist', whitelistAnom(F.col('uuid'),F.hour(F.col('last_seen')))

print('# Whitelist cargada')
test.select('features','anomalia_whitelist').show()


# Anomalia
threshold = np.load('./data/thresholdBt.npy',allow_pickle=True)
limit = 0


test = test.withColumn('anomalia', F.when(test.anomalia_whitelist == True, True).otherwise(test.anomalia_model))
test.select('features','anomalia').show()






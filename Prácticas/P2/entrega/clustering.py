# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering#,estimate_bandwidth
from sklearn.cluster import MeanShift,DBSCAN, MiniBatchKMeans
from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns
from scipy.cluster import hierarchy


from sklearn import cluster

random_seed = 54142189

################### FUNCIONES ###########################

def getPrediction(algorithm, X):
    t = time.time()
    cluster_predict = algorithm.fit_predict(X) 
    tiempo = time.time() - t

    return cluster_predict, tiempo



def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


# Función para pintar Scatter Matrix 
def ScatterMatrix(X_model, name, caso):
    sns.set()
    variables = list(X_model)
    variables.remove('cluster')
    sns_plot = sns.pairplot(data=X_model, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="auto")
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    direct = 'img/scatter/Caso'+str(caso)+'-'
    plt.savefig(direct + name + ".png")
    plt.clf()
    
    

# Función para pintar heatmap
def Heatmap(X_model, name, caso):
    meanDF = X_model.groupby("cluster").mean()
    sns.heatmap(data=meanDF, linewidths=.1, cmap="Blues", annot=True, xticklabels='auto')
    plt.xticks(rotation=0)
    direct = 'img/heatmap/Caso'+str(caso)+'-'
    plt.savefig(direct + name + ".png")
    plt.clf()
    



#Calculamos el dendograma en el jerárquico
def Dendrogram(X, name,random_seed, caso):
        
    #linkage_array = ward(X_normal)
    #dendrogram(linkage_array,leaf_rotation=0., leaf_font_size=5.)
    #plt.savefig('img/dendogram/' + name + ".png")
    #plt.clf()
    
    #No puedo tener muchos elementos.
    #Hago un muestreo aleatorio para quedarme solo con 1000.
    if len(X)>1000:
        X = X.sample(1000, random_state=random_seed)
    
    #En clustering hay que normalizar para las métricas de distancia
    X_normal = preprocessing.normalize(X, norm='l2')

    algorithm = cluster.AgglomerativeClustering(n_clusters=100, linkage='ward')            
            
    
    cluster_predict, tiempo = getPrediction(algorithm, X_normal)
  
    
    k = len(set(cluster_predict))
    print(": k: {:3.0f}, ".format(k),end='')
    print("{:6.2f} segundos".format(tiempo))
    
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

    X_cluster = pd.concat([X, clusters], axis=1)
    
    min_size = 10
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
    k_filtrado = len(set(X_filtrado['cluster']))
    print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
    X_filtrado = X_filtrado.drop('cluster', 1)
            
            
    X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')

    
    linkage_array = hierarchy.ward(X_filtrado_normal)
    plt.figure(1)
    
    hierarchy.dendrogram(linkage_array,orientation='left')
    dendro = sns.clustermap(X_filtrado, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)    
    
    direct = 'img/dendogram/Caso'+str(caso)+'-'
    dendro.savefig(direct + name + ".png")
    plt.clf()


    
        
    






def CalcularMetricas(X_normal, cluster_predict):
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)

    #el cálculo de Silhouette puede consumir mucha RAM. 
    #Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, 
    #p.ej., el 20%
    muestra_silhoutte = 0.2 if (len(X_normal) > 10000) else 1.0

        
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, 
        metric='euclidean',sample_size=floor(muestra_silhoutte*len(X_normal)), random_state=random_seed)

    return metric_CH, metric_SC



def Preprocesado(data):
    #Se pueden reemplazar los valores desconocidos por un número
    #datos = datos.replace(np.NaN,0)
    
    #Imputamos con la media      
    for col in data:
       data[col].fillna(data[col].mean(), inplace=True)
    
    return data




def PrepararEstudio(caso):
    print("Leyendo el conjunto de datos...")
   
    datos = Preprocesado(pd.read_csv('mujeres_fecundidad_INE_2018.csv'))
    print("Fichero cargado correctamente")


    if(caso == 1):
        subset = datos.loc[(datos['DIFICULTAD']!=1)]
        usadas = ['EDADESTABLE' ,'ESTUDIOSA', 'TEMPRELA', 'NTRABA', 'MAMPRIMHIJO']
        
    elif(caso == 2):
        subset = datos.loc[(datos['EDAD']<40) & (datos['MODAANTICONCEP']!=5) & (datos['MODAANTICONCEP']!=14)]
        usadas = ['PRACTICANTE','NEMBANT', 'NHIJOBIO', 'NPARANT', 'NDESEOHIJO']
        
    elif(caso == 3):
        #Mujeres jóvenes
        subset = datos.loc[(datos['EDAD']<=28)]
        usadas = ['INGREHOG_INTER', 'TEMPRELA', 'EDINDECONO', 'EDAD', 'ESTUDIOSA']

    else:
        print("Por favor, introduzca un caso de estudio válido")

    X = subset[usadas]
    print(X)
    X_normal = X.apply(norm_to_zero_one)

    return (X, X_normal, usadas)



# Mostrar correlación entre variables
def Correlation(X):
    correlation = X.corr()
    sns.heatmap(correlation, square = True)
    plt.show()
    
    
def kplot(X, name, k, usadas,caso):
  print("\nGenerando kplot...")
  n_var = len(usadas)
  fig, axes = plt.subplots(k, n_var, sharey=False, figsize=(15,10))
  fig.subplots_adjust(wspace=0.2)
  colors = sns.color_palette(palette=None, n_colors=k, desat=None)
  
  for i in range(k):
    dat_filt = X.loc[X['cluster']==i]
    for j in range(n_var):
      sns.kdeplot(dat_filt[usadas[j]], shade=True, color=colors[i], ax=axes[i,j])
  
  plt.savefig('img/kplot/' +name+"_" + str(caso)+ "Kplot.png")
  plt.clf()
      



def box_plot(X, name, k, usadas, caso):
    n_var = len(usadas)
    fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(15, 15))
    fig.subplots_adjust(wspace=0.1)
    colors = sns.color_palette(palette=None, n_colors=k, desat=None)
    rango = []
    
    for i in range(n_var):
        rango.append([X[usadas[i]].min(), X[usadas[i]].max()])

    for i in range(k):
        dat_filt = X.loc[X['cluster']==i]
        for j in range(n_var):
            ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], ax=axes[i, j])
            ax.set_xlim(rango[j][0], rango[j][1])
    plt.savefig('img/boxplot/' +name+"_" + str(caso)+ "boxplot.png")
    plt.clf()      




def AlgoritmosPersonalizados(n, random_seed):
    meanshift = MeanShift(bin_seeding=True, cluster_all=True)
    k_means = KMeans(init='k-means++', n_clusters=n, n_init=5, random_state=random_seed)
    wardAgglo = AgglomerativeClustering(n_clusters=n, linkage='ward')
    miniBatchKMeans = MiniBatchKMeans(init='k-means++',n_clusters=n, n_init=5, max_no_improvement=10, verbose=0, random_state=random_seed)
    dbscan = DBSCAN(eps=0.2)
    
    
    algorithms = [("MeanShift", meanshift),
                  ("KMeans", k_means),
                  ("AgglomerativeClustering" ,wardAgglo),
                  
                  ("MiniBatchKM" ,miniBatchKMeans),
                  ("DBSCAN" ,dbscan)]
    
    return algorithms




def ClusteringAlgorithms(algorithms, X, X_normal, caso, usadas):

    random_seed = 54142189
    
    f = open("caso_" + str(caso) + ".txt", 'w')

    #center_algorithms = ('MiniBatchKM','MeanShift', 'KMeans')
    hierach_algorithms = ('AgglomerativeClustering')
    

    names = []
    num_cluster = []
    m_CH = []
    m_SC = []
    tiempos = []
    
    
    

    print("\nCaso de estudio ", caso, ", tamaño: ", len(X))
    f.write("\nCaso de estudio " + str(caso) + ", tamaño: " + str(len(X)))

    for name_algorithm, algorithm in algorithms:

        f.write("\n*Algoritmo*: " + name_algorithm + "\n")    
        
        cluster_predict, tiempo = getPrediction(algorithm, X_normal)

        # Pasar las predicciones a dataFrame
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])



        f.write("\nTamaño de cada cluster:\n")
        size=clusters['cluster'].value_counts()

        for num,i in size.iteritems():
           print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
           f.write('%s: %5d (%5.2f%%)\n' % (num,i,100*i/len(clusters)))
        print()


        #Calculamos las métricas
        print("Calculando métricas CH y SC de " + name_algorithm)
        metric_CH, metric_SC = CalcularMetricas(X_normal, cluster_predict)
        print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')
        print("Silhouette Coefficient: {:.5f}".format(metric_SC))



        #Guardamos variables para comparar posteriormente
        m_CH.append(metric_CH)
        tiempos.append(tiempo)
        names.append(name_algorithm)   
        m_SC.append(metric_SC)
        num_cluster.append(len(set(cluster_predict)))
        

        # Se añade la asignación de clusters como columna a X
        X_cluster = pd.concat([X, clusters], axis=1)
        X_normal_cluster = pd.concat([X_normal, clusters], axis=1)
 
    
        k = len(set(cluster_predict))
        cclusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
        data_cluster = pd.concat([X, cclusters], axis=1)
            
        
        #los casos 4, 5 y 6 son modificaciones, por tanto no mostramos las gráficas
        if caso!=4 and caso!=5 and caso!=6:
            
            #Pintamos KPlot y BoxPlot
            kplot(data_cluster, name_algorithm, k, usadas,caso)
            box_plot(data_cluster, name_algorithm, k, usadas, caso);


            # Pintamos el heatmap
            Heatmap(X_cluster, name_algorithm, caso)
    
            # Pintamos el scatter matrix
            ScatterMatrix(X_cluster, name_algorithm, caso)


            f.write(X_normal_cluster.groupby("cluster").mean().to_string())
    
            # Si es algoritmo Hierach pintamos el dendograma
            if name_algorithm in hierach_algorithms:
                Dendrogram(X, name_algorithm, random_seed, caso)


    
         
    algorithm = pd.DataFrame(names, columns=['Algoritmo'])
    ncluster = pd.DataFrame(num_cluster, columns=['Número Clusters'])
    CH = pd.DataFrame(m_CH, columns=['CH'])
    SC = pd.DataFrame(m_SC, columns=['SH'])
    time = pd.DataFrame(tiempos, columns=['Tiempo'])


    results = pd.concat([algorithm, ncluster, SC, CH, time], axis=1)
    f.write("\nResultados: \n")
    f.write(results.to_string())
    f.close()


#Casos de estudio

#Caso 1
X1, X1_normal, usadas1 = PrepararEstudio(1)
#Caso 2
X2, X2_normal, usadas2 = PrepararEstudio(2)
#Caso 3
X3, X3_normal, usadas3 = PrepararEstudio(3)

#Podemos calcular la correlación
#for i in [X1,X2,X3]:
#    Correlation(i)
    

#Algoritmos 
random_seed = 54142189
n_clusters = 5

algorithms = AlgoritmosPersonalizados(n_clusters, random_seed)



#EJECUCIÓN DE ALGORITMOS Y LOS CASOS DE USO
ClusteringAlgorithms(algorithms, X1, X1_normal, 1, usadas1)
ClusteringAlgorithms(algorithms, X2, X2_normal, 2, usadas2)
ClusteringAlgorithms(algorithms, X3, X3_normal, 3, usadas3)

    
#MODIFICACIONES 
wardAgglo_5 = AgglomerativeClustering(n_clusters=5, linkage='ward')
wardAgglo_2 = AgglomerativeClustering(n_clusters=2, linkage='ward')

k_means_5 = KMeans(init='k-means++', n_clusters=5, n_init=5, random_state=random_seed)
k_means_2 = KMeans(init='k-means++', n_clusters=2, n_init=5, random_state=random_seed)

    
modificacion = [("Agglo_5", wardAgglo_5),
              ("Agglo_2", wardAgglo_2),
              ("KMeans_5", k_means_5),
              ("KMeans_2", k_means_2)]
    

ClusteringAlgorithms(modificacion, X1, X1_normal, 4, usadas1)
ClusteringAlgorithms(modificacion, X2, X2_normal, 5, usadas2)
ClusteringAlgorithms(modificacion, X3, X3_normal, 6, usadas3)
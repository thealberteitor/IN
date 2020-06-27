# -*- coding: utf-8 -*-
"""
Autor:
    Alberto Jesús Durán López
Fecha:
    Diciembre/2019
Competición:
    https://www.drivendata.org/competitions/57/nepal-earthquake/
    
    Inteligencia de Negocio
    Doble Grado en Ingeniería Informática y Matemáticas
    Universidad de Granada
"""
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb

#Para ver matriz correlación
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy

from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier

from mlxtend.classifier import EnsembleVoteClassifier


#------------------------------------------------------------------------------
 
def LecturaDatos():

    train_data = pd.read_csv('./dataInfo/nepal_earthquake_tra.csv')
    lab = pd.read_csv('./dataInfo/nepal_earthquake_labels.csv')
    test_data = pd.read_csv('./dataInfo/nepal_earthquake_tst.csv')
     
    
    return train_data,lab,test_data

#------------------------------------------------------------------------------
    

def ConvertCatToNum(data_training, data_test):
    mask = data_training.isnull()
    data_x_tmp = data_training.fillna(9999)
    data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
    data_x_nan = data_x_tmp.where(~mask, data_training)
    
    mask = data_test.isnull() #máscara para luego recuperar los NaN
    data_x_tmp = data_test.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
    data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
    data_x_tst_nan = data_x_tmp.where(~mask, data_test) #se recuperan los NaN

    return data_x_nan.values, data_x_tst_nan.values, np.ravel(data_labels.values)

#------------------------------------------------------------------------------

def ImputarStrategy(strat):
    imp = Imputer(missing_values='NaN', strategy=strat)
    imp = imp.fit(X)
    X_train_imp = imp.transform(X)
    imp = imp.fit(X_tst)
    X_tst_imp = imp.transform(X_tst)
    
    return X_tst_imp, X_train_imp

#------------------------------------------------------------------------------
    
def PreprocesadoDatos(prep,data_training, data_test, data_labels):
    #Borrar las columnas que no tengan correlacion con el resto

    print("  Borrando columnas...")
    
    
    mustdrop1 = ['building_id']
    
    mustdrop2 = ['building_id','has_superstructure_other', 'has_secondary_use_health_post',
                'has_secondary_use_institution', 'has_secondary_use_school',
                'has_secondary_use_industry', 'has_secondary_use_gov_office',
                'has_secondary_use_use_police','has_secondary_use_other',
                'has_secondary_use_rental','has_superstructure_cement_mortar_stone',
                'has_secondary_use_hotel','plan_configuration',
                'has_superstructure_rc_engineered', 'has_superstructure_stone_flag',
                'has_superstructure_rc_non_engineered', 'legal_ownership_status',
                'has_superstructure_adobe_mud','has_secondary_use_agriculture']
    
    
    mustdrop3 = ['building_id','has_superstructure_other', 'has_secondary_use_health_post',
                'has_secondary_use_institution', 'has_secondary_use_school',
                'has_secondary_use_industry', 'has_secondary_use_gov_office',
                'has_secondary_use_use_police','has_secondary_use_other',
                'has_secondary_use_rental','has_superstructure_cement_mortar_stone',
                'has_secondary_use_hotel','plan_configuration',
                'has_superstructure_rc_engineered', 'has_superstructure_stone_flag',
                'has_superstructure_rc_non_engineered', 'legal_ownership_status',
                'has_superstructure_adobe_mud','has_secondary_use_agriculture',
                'has_superstructure_mud_mortar_brick','has_secondary_use',
                'has_superstructure_bamboo','count_families']
    
    
    data_training.drop(labels=mustdrop1, axis=1,inplace = True)
    data_test.drop(labels=mustdrop1, axis=1,inplace = True)
    data_labels.drop(labels='building_id', axis=1,inplace = True)
    
    if prep==1:

        biyeccion = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 
                     'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 
                     'ñ':15,'o':16, 'p':17, 'q':18, 'r':19, 's':20, 't':21, 
                     'u':22, 'v':23, 'w':24, 'x':25, 'y':26, 'z':27}
        
        
        preprocesado1 = {"roof_type": biyeccion,
                         "land_surface_condition": biyeccion,
                         "position": biyeccion,
                         "other_floor_type": biyeccion,
                         "legal_ownership_status": biyeccion,
                         "foundation_type": biyeccion,  
                         "ground_floor_type": biyeccion,
                         "plan_configuration": biyeccion,
                       }
        data_training.replace(preprocesado1, inplace=True)
        data_test.replace(preprocesado1, inplace=True)
    
    
    if prep==2:
        preprocesado2 = {"roof_type": {'n': 1, 'q': 2, 'x': 3},
                         "land_surface_condition": {'n': 1, 'o': 2, 't':3},
                         "position": {'j': 1, 'o': 2, 's': 3, 't': 4},
                         "other_floor_type": {'j': 1, 'q': 2, 's': 3, 'x': 4},
                         "legal_ownership_status": {'a': 1, 'r': 2, 'v': 3, 'w': 4},         
                         "foundation_type": {'h': 1, 'i': 2, 'r': 3, 'u': 4, 'w': 5},               
                         "ground_floor_type": {'f': 1, 'm': 2, 'v': 3, 'x': 4, 'z': 5},
                         "plan_configuration": {'a': 1, 'c': 2, 'd': 3, 'f': 4, 'm': 5, 'n': 6, 'o': 7, 'q': 8, 's': 9, 'u': 10}
                       }
        data_training.replace(preprocesado2, inplace=True)
        data_test.replace(preprocesado2, inplace=True)
    
    if prep==3:
        # categorical features
        #cat_features = data_training.columns[data_training.dtypes == 'object']
        
        data_training = pd.get_dummies(data_training)
        data_test = pd.get_dummies(data_test)
        
        
        
      
    
    return data_training.values, data_test.values, np.ravel(data_labels.values)

#------------------------------------------------------------------------------
    
def GraficoComprobarVar(data_lab):

    print("Gráfico - Distribución Variable de clasificación: ")
    var_clasificacion = 'damage_grade'
    sns.countplot(var_clasificacion, data = data_lab)
    plt.rcParams["figure.figsize"] = (30,30)
    plt.show()
    plt.clf()
    
    
    print("Diagrama de barras con damage_grade, clases")
    (data_labels.damage_grade.value_counts().sort_index().plot.bar(title="Buildings with Each Damage Grade"))

    
    print("La matriz de correlación ha de haberse mostrado")
    correlations = data_training.corr() 
    fig, ax = plt.subplots(figsize=(18,18)) 
    sns.heatmap(correlations, linewidths=0.125, ax=ax) 
    plt.savefig("corr.png") 
    plt.clf()
    
    print("Valores perdidos:")
    print(data_training.isnull().sum())
    data_training.isnull().sum().plot.bar()
    plt.show()
    plt.savefig("perdidos.png")
    plt.clf()
    
#------------------------------------------------------------------------------
#Validación cruzada con particionado estratificado y control de la aleatoridad fijando la semilla

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=54142189)
#skf = StratifiedShuffleSplit(n_splits=4, random_state=42, test_size=4)
#skf = KFold(n_splits=5, random_state=54)
def validacion_cruzada(modelo, X, y, cv):

    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (val): {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y[test],y_pred,average='micro') , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all

#------------------------------------------------------------------------------

def Algorithm(n):
    y_test = []
    
    if n==1:
        '''
        #####Submission 1 #######  F1-Score 0.6883
        Se trata de la configuración inicial por defecto, 
        con el preprocesado que venía y cross validation
        '''
        print("------ LightGBM...")
        lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=200,n_jobs=2)
        lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)
        
        devolver = lgbm
    
    
    if n==2:
        '''
        #####Submission 2 ####### F1-Score 0.6776
        Se trata de la configuración inicial por defecto, 
        con el preprocesado que venía y cross validation
        '''
        print("------ XGB...")
        xgbclf = xgb.XGBClassifier(n_estimators = 200,n_jobs=2)
        xgbclf, y_test_xgbclf = validacion_cruzada(xgbclf,X,y,skf)
        
        devolver = xgbclf
    
    
    
    
    if n==3:
        '''
        #####Submission 3 ####### F1-Score 0.6920
        Se ha ejecutado con el preprocesado inicial.
        No se ha usado cross-validation 
        '''
        rf1 = RandomForestClassifier(n_estimators=350,
                                criterion='gini',
                                max_features='auto',
                                max_depth=20, #16
                                min_samples_split=10,
                                oob_score=True,
                                random_state=54142189,
                                n_jobs=-1)
        devolver = rf1
    
    if n==4:
        '''
        #####Submission 4 ###### F1-Score 0.7024
        Probamos esta vez Catboost.
        Seguimos con la configuración inicial, es decir,
        usamos el preprocesado dado, pero a partir de las siguientes
        ejecutaciones usaremos otro debido a los malos resultados.
        No usamos cross-validation
        '''
        cbc= CatBoostClassifier(
                learning_rate=0.11,
                n_estimators=450,
                loss_function='MultiClass',
                eval_metric='TotalF1',
                od_pval=0.001,
                od_type='IncToDec',
                random_seed=54142189,
                bootstrap_type='MVS', #Probar Bayesian
                best_model_min_trees=250,
                max_depth=12)
        devolver = cbc
        
        
    if n==5: 
        '''
        #####Submission 5 ###### F1-Score 0.7081
        Se usa el preprocesado1, es decir, pasamos las variables numéricas 
        a categóricas. Para ello establecemos una biyección entre las letras
        del abecedario y los números naturales:
        a->1 , b->2 , c->3 .... z->27
        
        Se mejora algo más el algoritmo que usando el preprocesado inicial
        y pasa del 0.70 con amplio margen.
        '''
        rf1 = RandomForestClassifier(n_estimators=350,
                                criterion='gini',
                                max_features='auto',
                                max_depth=20, #16
                                min_samples_split=10,
                                oob_score=True,
                                random_state=54142189,
                                n_jobs=-1)
        devolver = rf1
    
    if n==6:
        '''
        #####Submission 6 ###### F1-Score 0.7360
        Se usa el preprocesado1, es decir, pasamos las variables numéricas 
        a categóricas. Para ello establecemos una biyección entre las letras
        del abecedario y los números naturales:
        a->1 , b->2 , c->3 .... z->27
        
        Se mejora bastante más que usando cualquier otro algoritmo
        '''
        cbc= CatBoostClassifier(
                learning_rate=0.1,
                n_estimators=550,
                loss_function='MultiClass',
                eval_metric='TotalF1',
                od_pval=0.001,
                od_type='IncToDec',
                random_seed=54142189,
                bootstrap_type='MVS', #Probar Bayesian
                best_model_min_trees=250,
                max_depth=12)
        devolver = cbc
    
    if n==7:
        ''' #####Submission 7 #### F1-score 0.6535
        Oversampling de las clases no ofrece buenos resultados.
        Usamos algoritmos que nos permitan indicar que las clases están desbalanceadas
        
        
        import smote_variants as sv  
        oversampler = sv.ProWSyn(proportion=1.0, n_neighbors=5, L=5, theta=1.0, n_jobs=-1, random_state=123)
        #oversampler = sv.kmeans_SMOTE(proportion=1.0, n_neighbors=3, n_jobs=-1, random_state=123)
        #oversampler = sv.polynom_fit_SMOTE(proportion=1.0, topology='star', random_state=123)
        X, y = oversampler.sample(X,y)
        '''
        cbc= CatBoostClassifier(
                learning_rate=0.1,
                n_estimators=550,
                loss_function='MultiClass',
                eval_metric='TotalF1',
                od_pval=0.001,
                od_type='IncToDec',
                random_seed=54142189,
                bootstrap_type='MVS', #Probar Bayesian
                best_model_min_trees=250,
                max_depth=12)
        devolver = cbc
    

    
    
    if n==8:
        '''
        #####Submission 8 #### f1 score 0.6887
        Probamos Random Forest asignándole los pesos manualmente,
        preprocesado 1, biyección números naturales.
        Para establecer los pesos realizamos un sistema de ec. lineales
        Sabemos que la clase 1, 2 y 3 se reparten en {7.5%, 52.5%, 40%} respect.
        
        x+y+z =1  |
        0.075*x=R  |   =>   x=0.7516,  y=0.107,   z=0.14
        0.525*y=R  |
        0.4*z=R    |
        '''
        rf1 = RandomForestClassifier(n_estimators=1050,
                                criterion='gini',
                                max_features='auto',
                                max_depth=20, #16
                                min_samples_split=10,
                                oob_score=True,
                                random_state=54142189,
                                n_jobs=-1,
                                #class_weight='balanced'
                                class_weight={1: 0.7516, 2: 0.107, 3: 0.14}
                                )
        devolver = rf1
    
    if n==9:
        '''
        #####Submission 9 #### f1-score-0.7444
        LightGBM tiene un parámetro para indicar que las clases están desbalanceadas
        is_unbalance=True, lo ejecutamos con 2000 árboles pues LightGBM es muy eficaz
        y termina en pocos minutos. 
        Usamos preprocesado 2, biyección con números naturales restringiendo el dominio
        '''
        
        lgbm = lgb.LGBMClassifier(objective='multiclass',
                                  is_unbalance=True,
                                  max_depth=20,
                                  n_jobs=-1,
                                  n_estimators = 2000,
                                  learning_rate=0.1)
        devolver = lgbm
        
        
        
    if n==10:  
        '''
        #####Submission 10### f1-score - 0.7454
        Obtengo 2950 estimators y 0.112 como configuración optima
        Ha tardado 24h en ejecutarse
        '''
        
        print("------ LightGBM + GRID...")
    

        lgbm = lgb.LGBMClassifier(objective='multiclass',
                                  is_unbalance=True,
                                  max_depth=20,
                                  n_jobs=-1)
        
        
        parametros = {
            
        "n_estimators": [2800,2850, 2900, 2950,
                          3000,3050, 3100, 3150,
                          3200,3250, 3300, 3350,
                          3400, 3450, 3500, 3550,
                          3600, 3650, 3700, 3750],
        
        "learning_rate": [0.111, 0.112, 0.114, 0.116, 0.118, 0.119,
                          0.121, 0.122, 0.124, 0.126, 0.128, 0.131]
        }
        
        

        devolver = GridSearchCV(estimator=lgbm,
                  param_grid=parametros,
                  scoring='f1_micro',#puede ser macro, weighted
                  cv=3,
                  n_jobs=-1)
        
        print("Obteniendo mejor configuración: ")
        devolver.fit(X,y)
        
        print(devolver.best_params_)
        
        mejorNE = devolver.best_params_['n_estimators']
        mejorLR = devolver.best_params_['learning_rate']

        
        print("Calculando algoritmo con mejores parámetros")
        best = lgb.LGBMClassifier(learning_rate=mejorLR,
                                  objective='multiclass',
                                  n_estimators=mejorNE,
                                  max_depth=20,
                                  n_jobs=-1,
                                  is_unbalance=True)
        
        devolver = best
    
    if n ==11:
        '''#####Submission 11 #### f1-score 0.7454
        Realizo cross validation con la configuración obtenida en el apartado 
        anterior y obtengo los mismos resutados
        '''
        print("LightGBM")
        best = lgb.LGBMClassifier(learning_rate=0.108,
                                  objective='multiclass',
                                  n_estimators=3198,
                                  max_depth=20,
                                  n_jobs=-1,
                                  is_unbalance=True)
        
        
        devolver = best
    
    if n==12:
        '''
        #####Submission 12 #### f1 score - 0.7466
        Obtengo 3198 y 0.108 como mejores parámetros. Ejecuto sobre esa configuración,
        obteniendo un rank = 81
        '''
        print("------ LightGBM + GRID...")
    
        lgbm = lgb.LGBMClassifier(objective='multiclass',
                                  max_depth=20,
                                  n_jobs=4)
        
        parametros = {
        "n_estimators": [3195, 3196, 3197, 3198, 3199,3200,
                         3205, 3206, 3207, 3208, 3209],
        
        "learning_rate": [0.106, 0.107, 0.108, 0.109, 0.11]
        }
        
        devolver = GridSearchCV(estimator=lgbm,
                  param_grid=parametros,
                  scoring='f1_micro',#puede ser macro, weighted
                  cv=3,
                  n_jobs=4)
        
        print("Obteniendo mejor configuración: ")
        devolver.fit(X,y)
        
        print(devolver.best_params_)
        
        mejorNE = devolver.best_params_['n_estimators']
        mejorLR = devolver.best_params_['learning_rate']
        
        print("----------------------------------")
        print("Learning rate óptimo: " + str(mejorLR))
        print("N_estimador óptimo: " + str(mejorNE))
        print("----------------------------------")

        
        print("Calculando algoritmo con mejores parámetros")
        best = lgb.LGBMClassifier(learning_rate=mejorLR,
                                  objective='multiclass',
                                  n_estimators=mejorNE,
                                  max_depth=20,
                                  n_jobs=4,
                                  is_unbalance=True)
    
        
        
    if n==13: 
        '''
        #####Submission13 #### f1-score 0.7466
        Realizo un grid potente para obtener los mejores parámetros
        (+1 día en ejecutar)
        
        se obtiene scale_pos_weight 0.5
        '''       
        
        print("------ LightGBM + GRID...")
    
        lgbm = lgb.LGBMClassifier(objective='multiclass', n_jobs=5)
        
        parametros = {
            "n_estimators": [ 3197, 3198, 3199, 3200],
            "learning_rate": [0.1075, 0.108, 0.1085], 
            "max_depth": [16, 18, 20, 22, 24],
            "scale_pos_weight": [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            #"is_unbalanced" : [True, False]
        }
        
        devolver = GridSearchCV(estimator=lgbm,
                  param_grid=parametros,
                  scoring='f1_micro',#puede ser macro, weighted
                  cv=3,
                  n_jobs=5)
        
        print("Obteniendo mejor configuración: ")
        devolver.fit(X,y)
        
        print(devolver.best_params_)
        
        mejorNE = devolver.best_params_['n_estimators']
        mejorLR = devolver.best_params_['learning_rate']
        mejordepth = devolver.best_params_['max_depth']
        scale_best = devolver.best_params_['scale_pos_weight']
        #is_un = devolver.best_params_['is_unbalanced']
    
        
        print("Calculando algoritmo con mejores parámetros")
        best = lgb.LGBMClassifier(n_estimators=mejorNE,
                                  learning_rate=mejorLR,
                                  max_depth=mejordepth,
                                  scale_pos_weight=scale_best,
                                  objective='multiclass',
                                  n_jobs=-1)
                                  #is_unbalance=is_un)
        devolver=best
        
    if n==14:
        '''
        #####Submission 14 #### f1 score - 0.7475
        Probamos bagging con nuestro mejor algoritmo
        '''        
        lgbm = lgb.LGBMClassifier(n_estimators=3198,
                                  objective='multiclass', 
                                  n_jobs=-1,
                                  learning_Rate=0.108,
                                  max_depth=20,
                                  num_leaves=31)
        
                
        bagfinal = BaggingClassifier(base_estimator=lgbm,
                                     n_estimators=20,
                                     #n_estimators=15,
                                     n_jobs=-1, 
                                     random_state=54142189)
    
        devolver = bagfinal
    
    if n==15:
        '''
        #####Submission 15 #### f1 score - 0.7478
        Probamos bagging con nuestro mejor algoritmo
        ''' 
        best = lgb.LGBMClassifier(learning_rate=0.108,
                                  objective='multiclass',
                                  n_estimators=3198,
                                  max_depth=20,
                                  n_jobs=-1,
                                  scale_pos_weight=0.0001)
        
        clf1 = BaggingClassifier(base_estimator=best,
                                     n_estimators=15,
                                     n_jobs=-1,
                                     random_state=54142189)
        
        devolver=clf1
    
    if n==16:
        '''#gana 20, pero is_unbalance=True lo empeora
        #####Submission 16 #### f1 score - 0.7475
        '''
        
        lgbm = lgb.LGBMClassifier(objective='multiclass',
                                  max_depth=20,
                                  n_jobs=5,
                                  n_estimators =3198,
                                  is_unbalance=True,
                                  learning_rate=0.108)
        
        bag = BaggingClassifier(base_estimator=lgbm, n_jobs=5, random_state=54142189)
        
        
        parametros = {
            "n_estimators": [10,15,20],
        }
        
        devolver = GridSearchCV(estimator=bag,
                  param_grid=parametros,
                  scoring='f1_micro',#puede ser macro, weighted
                  cv=3,
                  n_jobs=5)
        
        print("Obteniendo mejor configuración: ")
        
        devolver.fit(X,y)
        print(devolver.best_params_)
        mejorNE = devolver.best_params_['n_estimators']
        
        
        bagfinal = BaggingClassifier(base_estimator=lgbm, 
                                     n_estimators=mejorNE,
                                     n_jobs=5,
                                     random_state=54142189)
    
        devolver = bagfinal
    if n==17:
        '''
        ####Submission 17##### f1 score 0.7456
        tiene que haber undertraining u over
        '''
        lgbm = lgb.LGBMClassifier(objective='multiclassova',
                                  learning_rate=0.108,
                                  n_estimators=3198,
                                  max_depth=20,
                                  n_jobs=-1,
                                  random_state=54142189,
                                  class_weight= {1: 1, 2: 0.8, 3: 0.7}
                                  )
        devolver =  lgbm
        
    if n==18:
        '''
        ####Submission 18##### f1 score 0.7480
        '''
        
        lgbm = lgb.LGBMClassifier(learning_rate=0.108,
                                  objective='multiclass',
                                  n_estimators=3198,
                                  max_depth=20,
                                  n_jobs=-1)
        

        
        clf1 = BaggingClassifier(base_estimator=lgbm,
                                     n_estimators=20,
                                     n_jobs=-1, 
                                     bootstrap=True,
                                     random_state=54142189)
        
        
        clf2 = BaggingClassifier(base_estimator=lgbm,
                                     n_estimators=20,
                                     n_jobs=-1, 
                                     bootstrap=False,
                                     random_state=54142189)
        
        devolver = EnsembleVoteClassifier(clfs=[clf1, clf2],
                              weights=[1, 1], voting='soft')
    
    if n ==19:
        '''
        ####Submission 19#### f1 score 0.7463
        '''
        
        lgbm = lgb.LGBMClassifier(objective='multiclass',
                                  learning_rate=0.108,
                                  n_estimators=3198,
                                  max_depth=20,
                                  n_jobs=-1,
                                  random_state=54142189,
                                  class_weight= {1: 1, 2: 0.8, 3: 0.7}
                                  )
        devolver =  lgbm
    
    if n==20:
        '''
        ####Submission 20##### f1 score 0.7478
        print(" LightGBM")
        '''
    
        lgbm = lgb.LGBMClassifier(learning_rate=0.1035,  
                                  objective='multiclassova',
                                  n_estimators=3198,
                                  n_jobs=-1,
                                  max_depth=20,
                                  random_state=54142189)
            
    
        clf1 = BaggingClassifier(base_estimator=lgbm,
                                 n_estimators=20,
                                 n_jobs=-1, 
                                 bootstrap=True,
                                 random_state=54142189)
            
            
        clf2 = BaggingClassifier(base_estimator=lgbm,
                                 n_estimators=20,
                                 n_jobs=-1, 
                                 bootstrap=False,
                                 random_state=54142189)
            
            
        
        devolver = EnsembleVoteClassifier(clfs=[clf1, clf2],
                              weights=[1, 1], voting='hard')
    
    
    
    if n==21:
        
        '''
        ####Submission 21 ##### f1 score 0.7471
        Usamos oversampling con smote para disminuir overtraining
        smote distance
        '''
        print("------ LightGBM + GRID...")
    
        lgbm = lgb.LGBMClassifier(objective='multiclassova',
                                      n_estimators=3198,
                                      n_jobs=-1,
                                      max_depth=20,
                                      random_state=54142189)
        parametros = {
            "learning_rate" : [0.102, 0.104, 0.106, 0.108, 0.11],
            "num_leaves" : [32,33,34],
            "max_bin" : [350, 400,450]
        }

        devolver = GridSearchCV(estimator=lgbm,
                  param_grid=parametros,
                  scoring='f1_micro',#puede ser macro, weighted
                  cv=3,
                  n_jobs=-1)
        
        print("Obteniendo mejor configuración: ")
        devolver.fit(X,y)
        
        mejorLR = devolver.best_params_['learning_rate']
        mejorbin = devolver.best_params_['max_bin']
        mejorh = devolver.best_params_['num_leaves']
        
        print(devolver.best_params_)

        
        print("Calculando algoritmo con mejores parámetros")
        devolver = lgb.LGBMClassifier(objective='multiclassova',
                                      n_estimators=3198,
                                      n_jobs=-1,
                                      num_leaves=mejorh,
                                      max_bin=mejorbin,
                                      max_depth=20,
                                      #class_weight= {1: 1, 2: 0.8, 3: 0.7},
                                      random_state=54142189,
                                      learning_rate=mejorLR)
        
        
        devolver = lgb.LGBMClassifier(objective='multiclassova',
                                      n_estimators=3198,
                                      n_jobs=-1,
                                      num_leaves=34,
                                      max_bin=500,
                                      max_depth=20,
                                      random_state=54142189,
                                      learning_rate=0.1)
    

    if n==22:
        
        '''
        ####Submission 22 ##### f1 score 0.7478
        Usamos oversampling con smote para disminuir overtraining
        '''
        
        print("------ LightGBM")
        
        lgbm = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=35,
                                          max_bin=500,
                                          max_depth=24,
                                          random_state=54142189)
        
    if n==23:
        '''
        ####Submission 23 f1 score 0.7486
        '''
        lgbm = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=24,
                                          random_state=54142189)
        
        devolver = BaggingClassifier(base_estimator=lgbm,
                                     n_estimators=15,
                                     n_jobs=-1, 
                                     bootstrap=False,
                                     random_state=54142189)
        
    if n==24:
        '''
        ####Submissionf1-score 0.7484
        24 me ha fallado pq bootstrap false ha sobreentrenado
        '''
        
        lgbm = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=24,
                                          random_state=54142189)
        

        
        clf1 = BaggingClassifier(base_estimator=lgbm,
                                     n_estimators=15,
                                     n_jobs=-1, 
                                     bootstrap=True,
                                     random_state=54142189)
        
        
        clf2 = BaggingClassifier(base_estimator=lgbm,
                                     n_estimators=15,
                                     n_jobs=-1, 
                                     bootstrap=False,
                                     random_state=54142189)
        
        devolver = EnsembleVoteClassifier(clfs=[clf1, clf2],
                              weights=[1, 1], voting='soft')
        
    if n==25:
        '''
        ####Submission 25 smote distance
        '''
        lgbm1 = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=22,
                                          random_state=54142189)
        
        
        clf1 = BaggingClassifier(base_estimator=lgbm1,
                                     n_estimators=20,
                                     n_jobs=-1, 
                                     bootstrap=False,
                                     random_state=54142189)
        
        
        lgbm2 = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=24,
                                          random_state=54142189)
        
        
        clf2 = BaggingClassifier(base_estimator=lgbm2,
                                     n_estimators=15,
                                     n_jobs=-1, 
                                     random_state=54142189)
        
        #clf1 0.7485, clf2 0.7486
        devolver = EnsembleVoteClassifier(clfs=[clf1, clf2],
                              weights=[1, 1], voting='hard')
    if n==26:
        '''
        ####Submission 26 ,  soft pero con smote distance
        '''
        lgbm1 = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=22,
                                          random_state=54142189)
        
        
        clf1 = BaggingClassifier(base_estimator=lgbm1,
                                     n_estimators=20,
                                     n_jobs=-1, 
                                     bootstrap=False,
                                     random_state=54142189)
        
        
        lgbm2 = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=24,
                                          random_state=54142189)
        
        
        clf2 = BaggingClassifier(base_estimator=lgbm2,
                                     n_estimators=15,
                                     n_jobs=-1, 
                                     random_state=54142189)
        
        #clf1 0.7485, clf2 0.7486
        devolver = EnsembleVoteClassifier(clfs=[clf1, clf2],
                              weights=[1, 1], voting='soft')
        
        
    if n==27:
        '''
        ####Submission 27 , f1-score 0.7496
		#smote topology star
        '''
        
        lgbm1 = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=22,
                                          random_state=54142189)
        
        
        clf1 = BaggingClassifier(base_estimator=lgbm1,
                                     n_estimators=20,
                                     n_jobs=-1, 
                                     bootstrap=False,
                                     random_state=54142189)
        
        
        lgbm2 = lgb.LGBMClassifier(learning_rate=0.1,  
                                          objective='multiclassova',
                                          n_estimators=3198,
                                          n_jobs=-1,
                                          num_leaves=34,
                                          max_bin=500,
                                          max_depth=24,
                                          random_state=54142189)
        
        
        clf2 = BaggingClassifier(base_estimator=lgbm2,
                                     n_estimators=15,
                                     n_jobs=-1, 
                                     random_state=54142189)
        
        #clf1 0.7485, clf2 0.7486
        devolver = EnsembleVoteClassifier(clfs=[clf1, clf2],
                              weights=[1, 1], voting='soft')
        
        
    
        
    

    return devolver, y_test


#------------------------------------------------------------------------------



#Se convierten las variables categóricas a variables numéricas (ordinales)
#X , X_tst, y = ConvertCatToNum(data_training, data_test)

####################Programa principal###############################
    
#Leemos los datos
data_training, data_labels, data_test = LecturaDatos()


#Comprobando la distribución de la variable de clasificación + matriz correlación
#GraficoComprobarVar(data_labels)


#Se quitan las columnas que no se  usan apenas
X, X_tst, y = PreprocesadoDatos(3, data_training, data_test, data_labels)


#Sustituir el valor de n por el modelo que se desea ejecutar
n=1
    
print("##################################")
print("Ejecutando algoritmo número " + str(n))
print("##################################")
    
    
    
import smote_variants as sv
#oversampler = sv.ProWSyn(proportion=1.0, n_neighbors=5,L=5, theta=1.0, n_jobs=-1, random_state=2)
#oversampler= sv.MulticlassOversampling(sv.distance_SMOTE(random_state=2))
oversampler = sv.MulticlassOversampling(sv.polynom_fit_SMOTE(topology='star',random_state=2))
    
    
# X_sam and y_sam contain the oversampled dataset
X_sam, y_sam= oversampler.sample(X, y)
    
'''
import collections
height=[collections.Counter(y_sam)[1],collections.Counter(y_sam)[2],collections.Counter(y_sam)[3]]
print(height)
print(collections.Counter(y))
plt.bar( ['low damage','medium damage','high damage'], height,color=['blue', 'orange', 'green'])

colors = {'class1 - Smote Polynom/star':'blue', 'class2 - Smote Polynom/star':'orange', 'class3 - Smote Polynom/star':'green'}         
labels = list(colors.keys())
handles = [plt.Rectangle((1,0),-1,-1, color=colors[label]) for label in labels]
plt.legend(handles, labels)


plt.savefig("sstar.png")
#plt.show()
plt.clf()
'''



clf, y_testt = Algorithm(n)
        
clf = clf.fit(X_sam,y_sam)
            
y_pred_tra = clf.predict(X)
        
valor = f1_score(y,y_pred_tra,average='micro')
    
    
print("F1 score (tra): {:.4f}".format(valor))
        
y_pred_tst = clf.predict(X_tst)
        
df_submission = pd.read_csv('./dataInfo/nepal_earthquake_submission_format.csv')
        
y_pred_tst = y_pred_tst.astype(int)
df_submission['damage_grade'] = y_pred_tst
        
print("Creando archivo submission número " + str(n) )
df_submission.to_csv("submission" + str(valor) +"- " + str(n) +".csv", index=False)

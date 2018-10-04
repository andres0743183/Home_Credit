#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:49:41 2018

@author: andres
"""

import pandas as  pd
import h2o  
from h2o.estimators.random_forest import H2ORandomForestEstimator
import matplotlib.pyplot as plt
import seaborn as sns


variables_todas=['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1',
 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2',
 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

def tipificar_h2o(train):
    """
    Esta función lee un objeto data frame de h2o y convierte las columnas con menos 
    de 20 categorías en factores.
    """    
    A=train.as_data_frame()
    for col in train.col_names:
        if pd.crosstab(A[col],columns="frecuencia").shape[0]<=5:
           train[col]= train[col].asfactor()  

##Datos con información general de los clientes que tuvieron un crédito.
app_train=pd.read_csv("application_train.csv")
app_test=pd.read_csv("application_test.csv")
##TARGET el 1 son clientes que tuvieron un retraso en sus pago por lo menos una ves.

#Todos los créditos anteriores del cliente provistos por otras instituciones financieras que se informaron a 
#la Oficina de Crédito (para clientes que tienen un préstamo en nuestra muestra).
#Para cada préstamo en nuestra muestra, hay tantas filas como número de créditos que el cliente tuvo en 
#Credit Bureau antes de la fecha de la solicitud.
#bureau=pd.read_csv("bureau.csv")

#bureau_balance=pd.read_csv("bureau_balance.csv")

#cc_balance=pd.read_csv("credit_card_balance.csv")

#installments_payments=pd.read_csv("installments_payments.csv")

#POS_CASH_balance=pd.read_csv("POS_CASH_balance.csv")


#pd.crosstab(index=datos["REGION_RATING_CLIENT_W_CITY"],columns="frecuencia")

app_test.loc[17177,"REGION_RATING_CLIENT_W_CITY"]=1

#evaluacion[evaluacion["REGION_RATING_CLIENT_W_CITY"]==-1]["REGION_RATING_CLIENT_W_CITY"]

h2o.init(max_mem_size=14)
train = h2o.H2OFrame(app_train)
evaluacion_h2o = h2o.H2OFrame(app_test)

train["TARGET"]= train["TARGET"].asfactor()  

#train.types["AMT_INCOME_TOTAL"]

train["CODE_GENDER"]= train["CODE_GENDER"].asfactor()  
evaluacion_h2o["CODE_GENDER"]= evaluacion_h2o["CODE_GENDER"].asfactor()  

train["FLAG_OWN_CAR"]= train["FLAG_OWN_CAR"].asfactor()  
evaluacion_h2o["FLAG_OWN_CAR"]= evaluacion_h2o["FLAG_OWN_CAR"].asfactor()  

train["FLAG_OWN_REALTY"]= train["FLAG_OWN_REALTY"].asfactor()  
evaluacion_h2o["FLAG_OWN_REALTY"]= evaluacion_h2o["FLAG_OWN_REALTY"].asfactor() 

train["CNT_CHILDREN"]= train["CNT_CHILDREN"].asnumeric() 
evaluacion_h2o["CNT_CHILDREN"]= evaluacion_h2o["CNT_CHILDREN"].asnumeric() 

train["AMT_INCOME_TOTAL"]= train["AMT_INCOME_TOTAL"].asnumeric() 
evaluacion_h2o["AMT_INCOME_TOTAL"]= evaluacion_h2o["AMT_INCOME_TOTAL"].asnumeric() 

train["AMT_CREDIT"]= train["AMT_CREDIT"].asnumeric() 
evaluacion_h2o["AMT_CREDIT"]= evaluacion_h2o["AMT_CREDIT"].asnumeric() 

train["AMT_ANNUITY"]= train["AMT_ANNUITY"].asnumeric() 
evaluacion_h2o["AMT_ANNUITY"]= evaluacion_h2o["AMT_ANNUITY"].asnumeric() 

train["AMT_GOODS_PRICE"]= train["AMT_GOODS_PRICE"].asnumeric() 
evaluacion_h2o["AMT_GOODS_PRICE"]= evaluacion_h2o["AMT_GOODS_PRICE"].asnumeric() 

train["NAME_TYPE_SUITE"]= train["NAME_TYPE_SUITE"].asfactor()  
evaluacion_h2o["NAME_TYPE_SUITE"]= evaluacion_h2o["NAME_TYPE_SUITE"].asfactor() 

train["NAME_INCOME_TYPE"]= train["NAME_INCOME_TYPE"].asfactor()  
evaluacion_h2o["NAME_INCOME_TYPE"]= evaluacion_h2o["NAME_INCOME_TYPE"].asfactor() 

train["NAME_EDUCATION_TYPE"]= train["NAME_EDUCATION_TYPE"].asfactor()  
evaluacion_h2o["NAME_EDUCATION_TYPE"]= evaluacion_h2o["NAME_EDUCATION_TYPE"].asfactor() 

train["NAME_FAMILY_STATUS"]= train["NAME_FAMILY_STATUS"].asfactor()  
evaluacion_h2o["NAME_FAMILY_STATUS"]= evaluacion_h2o["NAME_FAMILY_STATUS"].asfactor() 

train["NAME_HOUSING_TYPE"]= train["NAME_HOUSING_TYPE"].asfactor()  
evaluacion_h2o["NAME_HOUSING_TYPE"]= evaluacion_h2o["NAME_HOUSING_TYPE"].asfactor() 

train["REGION_POPULATION_RELATIVE"]= train["REGION_POPULATION_RELATIVE"].asnumeric() 
evaluacion_h2o["REGION_POPULATION_RELATIVE"]= evaluacion_h2o["REGION_POPULATION_RELATIVE"].asnumeric() 

train["DAYS_BIRTH"]= train["DAYS_BIRTH"].asnumeric() 
evaluacion_h2o["DAYS_BIRTH"]= evaluacion_h2o["DAYS_BIRTH"].asnumeric() 

train["DAYS_EMPLOYED"]= train["DAYS_EMPLOYED"].asnumeric() 
evaluacion_h2o["DAYS_EMPLOYED"]= evaluacion_h2o["DAYS_EMPLOYED"].asnumeric() 

train["DAYS_REGISTRATION"]= train["DAYS_REGISTRATION"].asnumeric() 
evaluacion_h2o["DAYS_REGISTRATION"]= evaluacion_h2o["DAYS_REGISTRATION"].asnumeric() 

train["DAYS_ID_PUBLISH"]= train["DAYS_ID_PUBLISH"].asnumeric() 
evaluacion_h2o["DAYS_ID_PUBLISH"]= evaluacion_h2o["DAYS_ID_PUBLISH"].asnumeric() 

train["OWN_CAR_AGE"]= train["OWN_CAR_AGE"].asnumeric() 
evaluacion_h2o["OWN_CAR_AGE"]= evaluacion_h2o["OWN_CAR_AGE"].asnumeric() 

train["FLAG_MOBIL"]= train["FLAG_MOBIL"].asfactor() 
evaluacion_h2o["FLAG_MOBIL"]= evaluacion_h2o["FLAG_MOBIL"].asfactor() 

train["FLAG_EMP_PHONE"]= train["FLAG_EMP_PHONE"].asfactor() 
evaluacion_h2o["FLAG_EMP_PHONE"]= evaluacion_h2o["FLAG_EMP_PHONE"].asfactor() 

train["FLAG_WORK_PHONE"]= train["FLAG_WORK_PHONE"].asfactor() 
evaluacion_h2o["FLAG_WORK_PHONE"]= evaluacion_h2o["FLAG_WORK_PHONE"].asfactor() 

train["FLAG_CONT_MOBILE"]= train["FLAG_CONT_MOBILE"].asfactor() 
evaluacion_h2o["FLAG_CONT_MOBILE"]= evaluacion_h2o["FLAG_CONT_MOBILE"].asfactor() 

train["FLAG_PHONE"]= train["FLAG_PHONE"].asfactor() 
evaluacion_h2o["FLAG_PHONE"]= evaluacion_h2o["FLAG_PHONE"].asfactor() 

train["FLAG_EMAIL"]= train["FLAG_EMAIL"].asfactor() 
evaluacion_h2o["FLAG_EMAIL"]= evaluacion_h2o["FLAG_EMAIL"].asfactor() 

train["OCCUPATION_TYPE"]= train["OCCUPATION_TYPE"].asfactor() 
evaluacion_h2o["OCCUPATION_TYPE"]= evaluacion_h2o["OCCUPATION_TYPE"].asfactor() 

train["CNT_FAM_MEMBERS"]= train["CNT_FAM_MEMBERS"].asnumeric() 
evaluacion_h2o["CNT_FAM_MEMBERS"]= evaluacion_h2o["CNT_FAM_MEMBERS"].asnumeric() 

train["REGION_RATING_CLIENT"]= train["REGION_RATING_CLIENT"].asfactor() 
evaluacion_h2o["REGION_RATING_CLIENT"]= evaluacion_h2o["REGION_RATING_CLIENT"].asfactor() 

##Valor por revisar
train["REGION_RATING_CLIENT_W_CITY"]= train["REGION_RATING_CLIENT_W_CITY"].asfactor() 
evaluacion_h2o["REGION_RATING_CLIENT_W_CITY"]= evaluacion_h2o["REGION_RATING_CLIENT_W_CITY"].asfactor() 

train["WEEKDAY_APPR_PROCESS_START"]= train["WEEKDAY_APPR_PROCESS_START"].asfactor() 
evaluacion_h2o["WEEKDAY_APPR_PROCESS_START"]= evaluacion_h2o["WEEKDAY_APPR_PROCESS_START"].asfactor() 

train["HOUR_APPR_PROCESS_START"]= train["HOUR_APPR_PROCESS_START"].asfactor() 
evaluacion_h2o["HOUR_APPR_PROCESS_START"]= evaluacion_h2o["HOUR_APPR_PROCESS_START"].asfactor() 

train["REG_REGION_NOT_LIVE_REGION"]= train["REG_REGION_NOT_LIVE_REGION"].asfactor() 
evaluacion_h2o["REG_REGION_NOT_LIVE_REGION"]= evaluacion_h2o["REG_REGION_NOT_LIVE_REGION"].asfactor() 

train["REG_REGION_NOT_WORK_REGION"]= train["REG_REGION_NOT_WORK_REGION"].asfactor() 
evaluacion_h2o["REG_REGION_NOT_WORK_REGION"]= evaluacion_h2o["REG_REGION_NOT_WORK_REGION"].asfactor() 

train["LIVE_REGION_NOT_WORK_REGION"]= train["LIVE_REGION_NOT_WORK_REGION"].asfactor() 
evaluacion_h2o["LIVE_REGION_NOT_WORK_REGION"]= evaluacion_h2o["LIVE_REGION_NOT_WORK_REGION"].asfactor() 

train["REG_CITY_NOT_LIVE_CITY"]= train["REG_CITY_NOT_LIVE_CITY"].asfactor() 
evaluacion_h2o["REG_CITY_NOT_LIVE_CITY"]= evaluacion_h2o["REG_CITY_NOT_LIVE_CITY"].asfactor() 

train["REG_CITY_NOT_WORK_CITY"]= train["REG_CITY_NOT_WORK_CITY"].asfactor() 
evaluacion_h2o["REG_CITY_NOT_WORK_CITY"]= evaluacion_h2o["REG_CITY_NOT_WORK_CITY"].asfactor() 

train["LIVE_CITY_NOT_WORK_CITY"]= train["LIVE_CITY_NOT_WORK_CITY"].asfactor() 
evaluacion_h2o["LIVE_CITY_NOT_WORK_CITY"]= evaluacion_h2o["LIVE_CITY_NOT_WORK_CITY"].asfactor() 

train["ORGANIZATION_TYPE"]= train["ORGANIZATION_TYPE"].asfactor()  ##Se puede crear grupos
evaluacion_h2o["ORGANIZATION_TYPE"]= evaluacion_h2o["ORGANIZATION_TYPE"].asfactor() 

train["EXT_SOURCE_1"]= train["EXT_SOURCE_1"].asnumeric() 
evaluacion_h2o["EXT_SOURCE_1"]= evaluacion_h2o["EXT_SOURCE_1"].asnumeric() 

train["EXT_SOURCE_2"]= train["EXT_SOURCE_2"].asnumeric() 
evaluacion_h2o["EXT_SOURCE_2"]= evaluacion_h2o["EXT_SOURCE_2"].asnumeric() 

train["EXT_SOURCE_3"]= train["EXT_SOURCE_3"].asnumeric() 
evaluacion_h2o["EXT_SOURCE_3"]= evaluacion_h2o["EXT_SOURCE_3"].asnumeric() 

#pd.crosstab(index=datos["EXT_SOURCE_2"],columns="frecuencia")
#pd.crosstab(index=evaluacion["EXT_SOURCE_2"],columns="frecuencia")

train["APARTMENTS_MODE"]= train["APARTMENTS_MODE"].asnumeric() 
evaluacion_h2o["APARTMENTS_MODE"]= evaluacion_h2o["APARTMENTS_MODE"].asnumeric() 

train["LIVINGAPARTMENTS_MODE"]= train["LIVINGAPARTMENTS_MODE"].asnumeric() 
evaluacion_h2o["LIVINGAPARTMENTS_MODE"]= evaluacion_h2o["LIVINGAPARTMENTS_MODE"].asnumeric() 

train["APARTMENTS_AVG"]= train["APARTMENTS_AVG"].asnumeric() 
evaluacion_h2o["APARTMENTS_AVG"]= evaluacion_h2o["APARTMENTS_AVG"].asnumeric() 

train["BASEMENTAREA_AVG"]= train["BASEMENTAREA_AVG"].asnumeric() 
evaluacion_h2o["BASEMENTAREA_AVG"]= evaluacion_h2o["BASEMENTAREA_AVG"].asnumeric() 

train["COMMONAREA_AVG"]= train["COMMONAREA_AVG"].asnumeric() 
evaluacion_h2o["COMMONAREA_AVG"]= evaluacion_h2o["COMMONAREA_AVG"].asnumeric() 

train["ELEVATORS_AVG"]= train["ELEVATORS_AVG"].asnumeric() 
evaluacion_h2o["ELEVATORS_AVG"]= evaluacion_h2o["ELEVATORS_AVG"].asnumeric() 

train["FLOORSMIN_AVG"]= train["FLOORSMIN_AVG"].asnumeric() 
evaluacion_h2o["FLOORSMIN_AVG"]= evaluacion_h2o["FLOORSMIN_AVG"].asnumeric() 

train["LANDAREA_AVG"]= train["LANDAREA_AVG"].asnumeric() 
evaluacion_h2o["LANDAREA_AVG"]= evaluacion_h2o["LANDAREA_AVG"].asnumeric() 

train["LIVINGAPARTMENTS_AVG"]= train["LIVINGAPARTMENTS_AVG"].asnumeric() 
evaluacion_h2o["LIVINGAPARTMENTS_AVG"]= evaluacion_h2o["LIVINGAPARTMENTS_AVG"].asnumeric() 

train["NONLIVINGAPARTMENTS_AVG"]= train["NONLIVINGAPARTMENTS_AVG"].asnumeric() 
evaluacion_h2o["NONLIVINGAPARTMENTS_AVG"]= evaluacion_h2o["NONLIVINGAPARTMENTS_AVG"].asnumeric() 

train["NONLIVINGAREA_AVG"]= train["NONLIVINGAREA_AVG"].asnumeric() 
evaluacion_h2o["NONLIVINGAREA_AVG"]= evaluacion_h2o["NONLIVINGAREA_AVG"].asnumeric() 

train["BASEMENTAREA_MODE"]= train["BASEMENTAREA_MODE"].asnumeric() 
evaluacion_h2o["BASEMENTAREA_MODE"]= evaluacion_h2o["BASEMENTAREA_MODE"].asnumeric() 

train["COMMONAREA_MODE"]= train["COMMONAREA_MODE"].asnumeric() 
evaluacion_h2o["COMMONAREA_MODE"]= evaluacion_h2o["COMMONAREA_MODE"].asnumeric() 

train["LANDAREA_MODE"]= train["LANDAREA_MODE"].asnumeric() 
evaluacion_h2o["LANDAREA_MODE"]= evaluacion_h2o["LANDAREA_MODE"].asnumeric() 

train["NONLIVINGAPARTMENTS_MODE"]= train["NONLIVINGAPARTMENTS_MODE"].asnumeric() 
evaluacion_h2o["NONLIVINGAPARTMENTS_MODE"]= evaluacion_h2o["NONLIVINGAPARTMENTS_MODE"].asnumeric() 

train["NONLIVINGAREA_MODE"]= train["NONLIVINGAREA_MODE"].asnumeric() 
evaluacion_h2o["NONLIVINGAREA_MODE"]= evaluacion_h2o["NONLIVINGAREA_MODE"].asnumeric() 

train["BASEMENTAREA_MEDI"]= train["BASEMENTAREA_MEDI"].asnumeric() 
evaluacion_h2o["BASEMENTAREA_MEDI"]= evaluacion_h2o["BASEMENTAREA_MEDI"].asnumeric() 

train["COMMONAREA_MEDI"]= train["COMMONAREA_MEDI"].asnumeric() 
evaluacion_h2o["COMMONAREA_MEDI"]= evaluacion_h2o["COMMONAREA_MEDI"].asnumeric() 

train["LANDAREA_MEDI"]= train["LANDAREA_MEDI"].asnumeric() 
evaluacion_h2o["LANDAREA_MEDI"]= evaluacion_h2o["LANDAREA_MEDI"].asnumeric() 

train["LIVINGAPARTMENTS_MEDI"]= train["LIVINGAPARTMENTS_MEDI"].asnumeric() 
evaluacion_h2o["LIVINGAPARTMENTS_MEDI"]= evaluacion_h2o["LIVINGAPARTMENTS_MEDI"].asnumeric() 

train["NONLIVINGAPARTMENTS_MEDI"]= train["NONLIVINGAPARTMENTS_MEDI"].asnumeric() 
evaluacion_h2o["NONLIVINGAPARTMENTS_MEDI"]= evaluacion_h2o["NONLIVINGAPARTMENTS_MEDI"].asnumeric() 

train["NONLIVINGAREA_MEDI"]= train["NONLIVINGAREA_MEDI"].asnumeric() 
evaluacion_h2o["NONLIVINGAREA_MEDI"]= evaluacion_h2o["NONLIVINGAREA_MEDI"].asnumeric() 

train["APARTMENTS_MEDI"]= train["APARTMENTS_MEDI"].asnumeric() 
evaluacion_h2o["APARTMENTS_MEDI"]= evaluacion_h2o["APARTMENTS_MEDI"].asnumeric() 


#evaluacion_h2o.types["TARGET"]

#train["EXT_SOURCE_1"]=train["EXT_SOURCE_1"].asnumeric()

#tipificar_h2o(train)
#tipificar_h2o(evaluacion_h2o)

splits = train.split_frame(ratios=[0.7], seed=123)   

mod=H2ORandomForestEstimator(nfolds=5,keep_cross_validation_predictions=True,max_depth=5,min_rows=20,ntrees=50)
mod.train(x=list(set(variables_todas) - set(['SK_ID_CURR', 'TARGET'])), y="TARGET", training_frame=splits[0])

#mejor_variables=["EXT_SOURCE_3","APARTMENTS_MODE","EXT_SOURCE_2","LIVINGAPARTMENTS_MODE","ORGANIZATION_TYPE",
     #            "YEARS_BUILD_MODE","YEARS_BUILD_AVG","YEARS_BUILD_MEDI","OCCUPATION_TYPE"]

#mod.train(x=mejor_variables, y="TARGET", training_frame=splits[0])

prediccion=mod.predict(evaluacion_h2o).as_data_frame()
mod.model_performance(splits[1]).plot()

mod.varimp()
mod.varimp_plot()


datos['TARGET'].value_counts()
mod.varimp()
mod.confusion_matrix()
mod.model_performance(splits[1]).confusion_matrix()

train.col_names

mod.model_performance(splits[1]).plot()

mod.predict()
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns



# Set the style of plots
plt.style.use('fivethirtyeight')

# Plot the distribution of ages in years
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
missing_values = missing_values_table(datos)
missing_values.head(20)
datos.dtypes.value_counts()
datos.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
(datos['DAYS_BIRTH'] / -365).describe()
correlations = datos.corr()['TARGET'].sort_values()






























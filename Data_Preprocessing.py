import numpy as np
import pandas as pd

## Clean unmatched IDs from excels
def GetCleanData():
    #Depressions measurements
    # Read the xlsx file
    data3 = pd.read_excel('Data/3_dependant_GenetikaExcercise_Confidential.xlsx')
    data3 = pd.get_dummies(data3)
    np_data = np.array(data3.values)

    # Medicines matrix
    # Read the xlsx file
    data2 = pd.read_excel('Data/2_symptoms_GenetikaExercise_Confidential.xlsx')
    data2 = pd.get_dummies(data2)
    np_data2 = np.array(data2.values)

    # Demographics matrix
    # Read the xlsx file
    data1 = pd.read_excel('Data/1_demographics_GenetikaExercise_Confidential.xlsx')
    data1 = pd.get_dummies(data1)
    np_data1 = np.array(data1.values)


    index_delete = []
    for index,val in enumerate(np_data2):
        if val[0] not in np_data[:,0]:
            index_delete.append(index)
            print(index)
    data2 = data2.drop(index_delete)
    np_data2 = np.delete(np_data2,index_delete,0)


    index_delete = []
    for index,val in enumerate(np_data):
        if val[0] not in np_data2[:,0]:
            index_delete.append(index)
            print([index , val[0]])
    np_data = np.delete(np_data,index_delete,0)
    data3 = data3.drop(index_delete)

    index_delete = []
    for index,val in enumerate(np_data1):
        if val[0] not in np_data2[:,0]:
            index_delete.append(index)
            print([index , val[0]])
    np_data1 = np.delete(np_data1,index_delete,0)
    data1 = data1.drop(index_delete)
    row_negative = data2.select_dtypes(include=[np.number]).ge(0).all(1)
    data2 = data2.drop(row_negative[row_negative==False].index[0])
    data1 = data1.drop(row_negative[row_negative==False].index[0])

    return data3,data2,data1 , row_negative


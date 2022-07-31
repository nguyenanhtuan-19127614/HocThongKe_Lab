import numpy as np 
import pandas as pd
import json
from LogisticRegression import  logistic_regression as lr

def map_feature(x1, x2):
#   Returns a new feature array with more features, comprising of 
#   x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2, etc.

    degree = 6
    out = np.ones([len(x1), int((degree + 1) * (degree + 2) / 2)])
    idx = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            a1 = x1 ** (i - j)
            a2 = x2 ** j
            out[:, idx] = a1 * a2
            idx += 1

    return out


#main program
def main():
    """READ DATA"""
    df = pd.read_csv("training_data.txt",
                     sep=",",
                     header=None,
                     names=("feature1","feature2","label"))
    #Tạo thêm giá trị cho x
    out = map_feature(df['feature1'], df['feature2'])

    X= out
    Y=df.iloc[:,[2]].values
    one=np.ones((len(X),1))
    x=np.concatenate((one,X),axis=1)

    #Khởi tạo lại trọng số weight
    w_shape = (X.shape[1]+1,1)
    w = np.ones(shape=w_shape).reshape(-1,1)

    """READ CONFIG"""
    CONFIG_FILE_PATH = "config.json"

    with open(CONFIG_FILE_PATH, "r") as configFile:
        configRaw = configFile.read()
        configJson = json.loads(configRaw)

    _alpha = configJson['Alpha']
    _lambda = configJson['Lambda']
    _numIter = configJson['NumIter']

    """CREATE MODEL"""
    model = lr.LogisticRegression(_alpha=_alpha,
                                  _lambda=_lambda,
                                  _iter=_numIter)

    w,cost_hs = model.training(feature=x,
                               label=Y,
                               weight=w)

    #Lưu 3 trọng số weight tốt nhất
    modelParams=pd.DataFrame({
        'W0':w[0],
        'W1':w[1],
        'W2':w[2]
    })
    modelParams.to_json(path_or_buf='model.json')

    #Tinh đo chinh xac
    accuracy=0.0

    #Confusion Matrix
    #True Positive, False Positive, True Negative, False Negative
    TP, FP, TN, FN = 0,0,0,0

    pre= model.predict(x,w)
    for i in range(0,len(pre)):

        if pre[i]<0.5:
            pre[i]=0

            if Y[i] == 0:
                TN += 1
            else:
                FN += 1

        else:
            pre[i]=1

            if Y[i] == 1:
                TP += 1
            else:
                FP += 1

        if pre[i]==Y[i]:
            accuracy+=1

    accuracy=accuracy*100/len(Y)

    #Precision = TruePositives / (TruePositives + FalsePositives)
    precision = TP / (TP + FP)

    #Recall = TruePositives / (TruePositives + FalseNegatives)
    recall = TP/ (TP + FN)

    #F1-Score = (2 * Precision * Recall) / (Precision + Recall)
    f1_score = (2 * precision * recall) / (precision + recall)
    #Lưu Accuracy
    accurData=pd.DataFrame({
        'accuracy':[accuracy],
        'precision':[precision],
        'recall': [recall],
        'f1_score': [f1_score]
    })
    accurData.to_json(path_or_buf='accuracy.json')

main()







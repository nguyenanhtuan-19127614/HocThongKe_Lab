
import numpy as np

#Class Logistic Regression
class LogisticRegression:

    def __init__(self, _alpha, _lambda, _iter):

        self._alpha = _alpha
        self._lambda = _lambda
        self._iter = _iter


    def sigmoid(self,z):

        return 1.0/(1+np.exp(-z))

    #Hàm Predict
    def predict(self,feature, weight):

        z=np.dot(feature,weight)
        return self.sigmoid(z)

    #hàm tính cost dựa trên 2 phương trình đề bài
    def compute_cost(self,feature, labels,weight):

        n = len(labels)
        pre = self.predict(feature,weight)

        cost_1=-(labels*np.log(pre))
        cost_2=-(1-labels)*np.log(1-pre)

        cost=cost_1+cost_2

        return cost.sum()/n

    #hàm tính gradient
    def compute_gradient(self, feature, labels, weight):

        n = len(labels)

        pre = self.predict(feature, weight)

        gradient = np.dot(feature.T, pre - labels)
        gradient = gradient / n

        return gradient

    #Gradient Descent
    def gradient_descent(self,feature,label,weight,learning_rate):


        gradient = self.compute_gradient(feature=feature,
                                         labels=label,
                                         weight=weight)

        weight=weight-gradient*learning_rate

        return weight

    # training
    def training(self,feature,label,weight):

        cost_hs=[]

        for i in range(int(self._iter)):

            weight = self.gradient_descent(feature, label, weight, self._alpha)
            cost = self.compute_cost(feature, label, weight)
            cost_hs.append(cost)

        return weight,cost_hs
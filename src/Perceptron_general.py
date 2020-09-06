"""
@Author: Jonatas Travessa
@Date: 06/09/2020
"""


import numpy as np


class Perceptron_general(object):
    def __init__(self, theta, learning_rate, bias):
        self.theta = theta
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = np.array([0.0, 0.0, 0.0])
        self.weights_adjusts = 0
        self.epochs = 0
        

    def __calc_sum(self, weights, example):
        return weights[0]*self.bias + weights[1]*example[0] + weights[2]*example[1]


    def __activ_step_function(self, u):
        if u >= self.theta:
            return 1.0
        else:
            return 0.0


    def __weight_adjust(self, weights, example, fu):
        return weights + self.learning_rate*(example[2] - fu)*np.array([self.bias, example[0], example[1]])
        

    def inittial_weights(self, lower_bound, superior_bound):
        self.weights = np.random.uniform(lower_bound, superior_bound, size=(3, ))
        self.weights = np.round(self.weights, 4)
        return self.weights


    # função para ser usada na parte 1
    def fitParte1(self, data):
        u = 0
        erros_in_epoch = 1000

        while (erros_in_epoch != 0):
            self.epochs += 1
            adjust_in_epoch = 0
            erros_in_epoch = 0
            
            for example in data:
            
                u = self.__calc_sum(self.weights, example)
                fu = self.__activ_step_function(u)
                
                if  fu != example[2]:
                    adjust_in_epoch += 1
                    erros_in_epoch += 1
                    self.weights_adjusts += 1

                    self.weights = self.__weight_adjust(self.weights, example, fu)
                    self.weights = np.round(self.weights, 4)

                    print("Vetor de pesos ajustado, novo vetor de pesos")
                    np.set_printoptions(precision=4, suppress=True)
                    print(self.weights)
                    print("")
                    
            print("Número de ajustes do vetor de pesos na época", self.epochs, ":", adjust_in_epoch)
            print("")
            
        print("Número total de ajustes do vetor de pesos:", self.weights_adjusts)


    # função para ser usada na parte 2
    def fitParte2(self, data):
        u = 0
        erros_in_epoch = 1000

        while (erros_in_epoch != 0):
            self.epochs += 1
            adjust_in_epoch = 0
            erros_in_epoch = 0
            
            for example in data:
            
                u = self.__calc_sum(self.weights, example)
                fu = self.__activ_step_function(u)
                
                if  fu != example[2]:
                    adjust_in_epoch += 1
                    erros_in_epoch += 1
                    self.weights_adjusts += 1

                    self.weights = self.__weight_adjust(self.weights, example, fu)
                    self.weights = np.round(self.weights, 4)

                
    def getWeights(self):
        return self.weights


    def getEpochs(self):
        return self.epochs
    

    def getWeightsAdjusts(self):
        return self.weights_adjusts

    
    def reset(self):
        self.weights = np.array([0.0, 0.0, 0.0])
        self.weights_adjusts = 0
        self.epochs = 0

        
    
    
    
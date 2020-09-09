"""
@Author: Dayvson Silva
@Date: 06/09/2020
"""
import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, epoch=100, learning_rate=np.random.uniform(0.1, 0.4, 1), baias=-1, random_train_set=False):
        self.epoch = epoch
        self.learning_rate= learning_rate
        self.weights = np.random.uniform(-0.5, 0.5, no_of_inputs + 1)
        self.baias = baias
        self.random_train_set = random_train_set
    
    def __add_baias(self, inputs):
        new_inputs = []
        for row in inputs:
            new_row = np.insert(row, 0, self.baias)
            new_inputs.append(new_row)
        return np.array(new_inputs)

    def __weight_adjust(self, x_train, y_predicto, y_real):
        erro = y_predicto - y_real
        self.weights = self.weights - self.learning_rate * erro * x_train

    def __activ_func(self, u): 
        if u >= 0.0:
            return 1.0
        else:
            return 0.0

    def __error(self, y_predicto, y_real):
        if y_predicto != y_real:
            return True
        else:
            return False

    def __random_sets(self, x_train, y_train):
        # Randomizando os dados de treino
        index = np.arange(len(x_train))
        np.random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]
        return x_train, y_train
        #---------------------------------
    
    def predict(self, x_test):
        x_test = self.__add_baias(x_test)
        y_predicto = []
        for _x_test in x_test:
            u = np.sum(np.dot(_x_test, self.weights))
            y = self.__activ_func(u)
            y_predicto.append(y)
        return np.array(y_predicto)

    def fit(self, x_train, y_train):
        no_erro = False
        epoch = 1
        count_adjust = 0
        x_train = self.__add_baias(x_train)
        while no_erro != True and epoch != self.epoch:
            print("\n\tÉpoca:", epoch)
            print("Pesos:", self.weights)
            print("Quantidade de Ajustes:", count_adjust)
            no_erro = True

            if self.random_train_set:
                x_train, y_train = self.__random_sets(x_train, y_train)

            for x, y in zip(x_train, y_train):
                u = np.sum(np.dot(x, self.weights))
                y_predicto = self.__activ_func(u)
                #print(y_predicto, y)
                if self.__error(y_predicto, y):
                    self.__weight_adjust(x, y_predicto, y)
                    no_erro = False
                    count_adjust += 1
            epoch += 1
        print("\n\nQuantidade total de épocas:", epoch)
        print("Quantidade total de ajustes:", count_adjust)
        print("Vetor final de pesos:", self.weights)
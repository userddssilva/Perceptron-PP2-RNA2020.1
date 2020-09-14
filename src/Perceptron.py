"""
@Author: Dayvson Silva
@Date: 06/09/2020
"""
import numpy as np

class Perceptron(object):
    
    #TODO: create class documentation

    def __init__(self, no_of_inputs, epoch=100, 
                    learning_rate=np.random.uniform(0.1, 0.4, 1), 
                    baias=-1, 
                    random_train_set=False, 
                    show_training=False,
                    weights=(-5.0, 0.5),
                    use_epoch=False):
        self.epoch = epoch
        self.learning_rate= learning_rate
        self.weights = np.random.uniform(weights[0], 
                                            weights[1], 
                                            no_of_inputs + 1)
        self.baias = baias
        self.random_train_set = random_train_set
        self.show_training = show_training
        self.use_epoch = use_epoch
    
    def __add_baias(self, inputs):
        new_inputs = []
        for row in inputs:
            new_row = np.insert(row, 0, self.baias)
            new_inputs.append(new_row)
        return np.array(new_inputs)

    def __weight_adjust(self, x_train, y_predicto, y_real):
        erro = y_predicto - y_real
        self.weights -= self.learning_rate * erro * x_train
        self.weights = np.round(self.weights, 4)

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

    def __show_training(self, epoch, count_adjust):
        print("\n\tÉpoca:", epoch)
        print("Pesos:", self.weights)
        print("Quantidade de Ajustes:", count_adjust)

    def __learning(self, x_train, y_train, no_erro):
        count_adjust = 0
        for x, y in zip(x_train, y_train):
            u = np.sum(np.dot(x, self.weights))
            y_predicto = self.__activ_func(u)
            if self.__error(y_predicto, y):
                self.__weight_adjust(x, y_predicto, y)
                no_erro = False
                count_adjust += 1
        return no_erro, count_adjust


    def __epochs(self, x_train, y_train):
        no_erro = False
        epoch = 1
        total_adjust = 0
        while no_erro != True:

            if self.use_epoch:
                if epoch == self.epoch:
                    break
                
            if self.random_train_set:
                x_train, y_train = self.__random_sets(x_train,
                                                         y_train)
            
            no_erro = True
            
            no_erro, count_adjust = self.__learning(x_train, 
                                                        y_train,
                                                        no_erro)

            if self.show_training:
                self.__show_training(epoch, count_adjust)

            epoch += 1
            total_adjust += count_adjust
        return epoch, total_adjust


    def fit(self, x_train, y_train):

        x_train = self.__add_baias(x_train)
        epoch, total_adjust = self.__epochs(x_train, y_train)
        
        print("\nQuantidade total de épocas:", epoch)
        print("Quantidade total de ajustes:", total_adjust)
        
        np.set_printoptions(precision=4, suppress=True)
        print("Vetor final de pesos:", np.round(self.weights, 4))
        return epoch, total_adjust
import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, epoch=100, learning_rate=0.01, baias=-1):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.random.random(no_of_inputs + 1)
        self.weights[0] = -1 #-> baias
        self.baias = baias
    
    def predict(self, inputs):
        u = np.dot(inputs, self.weights[1:])
        v = self.activation_function(u)
        return v

    def activation_function(self, u): 
        if u >= 0.5:
            return 1
        else:
            return 0

    def weight_adjust(self, x, _y, y):
        erro = _y - y
        self.weights[1:] = self.weights[1:] - self.learning_rate * erro * x

    def has_error(self, _y, y):
        if _y != y:
            return True
        else:
            return False
    
    def acumalite_sum(self):
        pass

    def train(self, training_inputs, labels):
        no_erro = False
        epoch = 1
        while no_erro != True and epoch != self.epoch:
            print("\tÉpoca:", epoch)
            print("Pesos:", self.weights[1:])
            no_erro = True
            for inputs, label in zip(training_inputs, labels):
                u = np.sum(np.dot(np.array(inputs), self.weights[1:]))+self.weights[0]
                _y = self.activation_function(u)
                print("Saída predita:", _y, "Saída real:", label)
                if self.has_error(_y, label):
                    self.weight_adjust(inputs, _y, label)
                    no_erro = False
            epoch += 1

def main():
    training_inputs = []
    training_inputs.append(np.array([1, 1]))
    training_inputs.append(np.array([1, 0]))
    training_inputs.append(np.array([0, 1]))
    training_inputs.append(np.array([0, 0]))
    labels = np.array([1, 1, 1, 0])

    perceptron = Perceptron(2, epoch=200)
    perceptron.train(training_inputs, labels)

    print(perceptron.weights)

    inputs = np.array([1, 1])
    #perceptron.predict(inputs) 
    #=> 1

    inputs = np.array([0, 1])
    #perceptron.predict(inputs) 
    #=> 0

if __name__ == "__main__":
    main()
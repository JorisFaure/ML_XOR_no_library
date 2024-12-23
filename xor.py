import numpy as np
import matplotlib.pyplot as plt


def squared_error(output_vec_predicted, output_vec_ref) : # (a(L) - y_ref)²
    error_vec = output_vec_predicted - output_vec_ref
    return np.dot(error_vec, error_vec)

def cost_derivative(output_vec_predicted, output_vec_ref) : # 2*(a(L) - y_ref)
    error_vec = output_vec_predicted - output_vec_ref
    return error_vec*2

def weights_derivative(previous_layer_values) : # a(L-1)
    return previous_layer_values

def cost_derivative(previous_layer_values, output_predicted, output_ref) : # grosse formule, voire cahier
    return weights_derivative(previous_layer_values) * sigmoid_derivative(val) * cost_derivative(output_predicted, output_ref)
        
        
# Pour la suite :
# - cost_derivative ca marchera pas, le sigmoid dépend du vecteur des valeurs des neurones qui est dans la class neuron.
# A la limite faut essayer de l'implem dans dans la classe nn ou neuron.
# Tu es en train d'essayer de faire la 1ere étape de backprop (calculer la dérivé de cout entre la layer output et la hidden layer.)
# Continuer à essayer de regarder la vidéo de 3 blues 1 brown sur le calculus
# TU galère à comprendre le passage en layer - 1.
#Essaye de redemander à chatgpt mtn que tu as mieux compris en génral.
    

class Neuron :
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1.0, 1.0, size = input_size)
        self.bias = 0
        self.value = 0
        
    def sigmoid(self, val) :
        return 1/(1 + np.exp(-val))
    
    def sigmoid_derivative(self, val) : # sig(z(L)) * (1 - sig(z(L)))
        return self.sigmoid(val) * (1 - self.sigmoid(val))
    
    def activate(self) :
        self.value = self.sigmoid(self.value)


class NeuralNetwork :
    def __init__(self, input_size, hidden_size, output_size) : #init neurones values, weight values and bias values
        self.hidden_neurons = [Neuron(input_size) for i in range (hidden_size)]
        self.output_neurons = [Neuron(hidden_size) for i in range (output_size)]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def print_values(self) :
        print("hidden neurons values =", [(neuron.value) for neuron in self.hidden_neurons])
        print("output neurons values =", [(neuron.value) for neuron in self.output_neurons])
        
    def print_biases(self) :
        print("hidden neurons biases =", [(neuron.bias) for neuron in self.hidden_neurons])
        print("output neurons biases =", [(neuron.bias) for neuron in self.output_neurons])
    
    def print_weights(self) :
        print("hidden neurons weights =", [(neuron.weights) for neuron in self.hidden_neurons])
        print("output neurons weights =", [(neuron.weights) for neuron in self.output_neurons])
        
    def feed_forward(self, input_vec) :
        for neuron in self.hidden_neurons : # we calculate the values of the hidden layer neurons
            neuron.value = np.dot(neuron.weights, input_vec) + neuron.bias
            neuron.activate()
            
        hidden_values = [neuron.value for neuron in self.hidden_neurons] #we retrieve the values of the hidden neurons to put them as input for output layer.

        for neuron in self.output_neurons : # we calculate the values of the output layer neurons
            neuron.value = np.dot(neuron.weights, hidden_values) + neuron.bias
            neuron.activate()
          
nn = NeuralNetwork(2, 4, 2)
nn.print_values()
nn.print_weights()
nn.print_biases()

input = np.array([[1], [1]])
nn.feed_forward(input)
nn.print_values()

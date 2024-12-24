import numpy as np
import matplotlib.pyplot as plt

class Neuron :
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1.0, 1.0, size = (input_size,))
        self.bias = 0
        self.value = 0
        self.activated_value = 0
        
    def sigmoid(self) :
        return 1/(1 + np.exp(-self.value))
    
    def sigmoid_derivative(self) : # sig(z(L)) * (1 - sig(z(L)))   : z(L) is the non-activated value
        return self.sigmoid() * (1 - self.sigmoid())
    
    def activate(self) :
        self.activated_value = self.sigmoid()
        
    def forward_pass(self, input_vec) :
        self.value = np.dot(self.weights, input_vec) + self.bias
        self.activate()
        
    def squared_error(self, output_ref) : # (y - y_ref)²
        error_value = self.activated_value - output_ref
        return error_value * error_value
    
    def cost_derivative_by_layer(self, output_ref) : # 2(y - y_ref)
        return 2*(self.activated_value - output_ref)
    
    def weights_derivative(self, input_vec) : # the input values
        return input_vec
    
    def update_weights_and_biases(self, input_vec, output_ref, lr) :
        bias_gradient = self.sigmoid_derivative() * self.cost_derivative_by_layer(output_ref) # (sig(z(L)))' * 2(y - y_ref)
        weight_gradient = input_vec * bias_gradient # input_values * (sig(z(L)))' * 2(y - y_ref)
        self.weights -= weight_gradient * lr
        self.bias -= bias_gradient * lr
        
        # Pour la hidden layer
        # en input vec je met l'input de base de mon réseau
        # en output_ref, je met le même output_ref utilisé dans la backprop de mon output_layer
        # la somme des gradient avec comme output_ref la valeur de chacune des sorties
        

class NeuralNetwork :
    def __init__(self, input_size, hidden_size, output_size, learning_rate) : #init neurones values, weight values and bias values
        self.hidden_neurons = [Neuron(input_size) for i in range (hidden_size)]
        self.output_neurons = [Neuron(hidden_size) for i in range (output_size)]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
    
    def print_values(self) :
        print("hidden neurons activated values =", [(neuron.activated_value) for neuron in self.hidden_neurons])
        print("output neurons activated values =", [(neuron.activated_value) for neuron in self.output_neurons])
        
    def print_biases(self) :
        print("hidden neurons biases =", [(neuron.bias) for neuron in self.hidden_neurons])
        print("output neurons biases =", [(neuron.bias) for neuron in self.output_neurons])
    
    def print_weights(self) :
        print("hidden neurons weights =", [(neuron.weights) for neuron in self.hidden_neurons])
        print("output neurons weights =", [(neuron.weights) for neuron in self.output_neurons])
        
    def forward_pass(self, input_vec) :
        for neuron in self.hidden_neurons : # we calculate the values of the hidden layer neurons
            neuron.forward_pass(input_vec)
            
        hidden_values = [neuron.activated_value for neuron in self.hidden_neurons] #we retrieve the values of the hidden neurons to put them as input for output layer.

        for neuron in self.output_neurons : # we calculate the values of the output layer neurons
            neuron.forward_pass(hidden_values)
    
    def backpropagation(self, output_vec) :
        hidden_values = np.array([neuron.activated_value for neuron in self.hidden_neurons])
        i = 0
        output_errors = []  # Stocke les gradients des biais pour la couche de sortie
        for i, neuron in enumerate(self.output_neurons) :
            neuron.update_weights_and_biases(hidden_values, output_vec[i], self.lr)
            
        for neuron in self.hidden_neurons :
            for h in range (self.output_size) :
                neuron.update_weights_and_biases(hidden_values, output_vec[h], self.lr)
                h+=1
    
    def train(self, input, output, epoch) :
        for i in range (epoch) :
            print(f"itération {i} :")
            for j, sample in enumerate(input) :
                # nn.print_weights()
                self.forward_pass(sample)
                # nn.print_weights()
                self.backpropagation(output[j])
                # nn.print_weights()
    
    def test(self, input, output) :
        for i, sample in enumerate(input) :
            nn.forward_pass(sample)
            for neurons in nn.output_neurons :
                print("predicted : ", neurons.activated_value, " | ref : ", output[i])
        
        
nn = NeuralNetwork(2, 2, 1, 0.05) #(input_size, hidden_size, output_size, lr)
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = np.array([[0], [1], [1], [0]])
nn.train(input, output, 1000)

nn.test(input, output)

# Ca ne marhe pas
#La backprop de la output marche bien
#La backrpop de la hidden marche pas, t'as pas trop capaté comment on faisais.
#Mais en gros tu devra refaire ta class Neuron, car pour la backprop de la hidden, tu dois avoir accès a des infos que t'aura pas accès si t'es uniquement dans la class neuron.
#Chatgpt a chaque fois il te mettait le calcul des gradient en dehors de la class Neuron (il mettait dans la class NN)
# Essaye de check dans le projet de dessin voire comment tu faisais.
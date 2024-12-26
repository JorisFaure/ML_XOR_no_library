import numpy as np
import matplotlib.pyplot as plt

class Neuron :
    def __init__(self, input_size): #init neurones values, weight values and bias values
        self.weights = np.random.uniform(-1.0, 1.0, size = (input_size,))
        self.bias = 0
        self.value = 0
        self.activated_value = 0
        
    def sigmoid(self) :
        return 1/(1 + np.exp(-self.value))
    
    def sigmoid_derivative(self) : # sig(z(L)) * (1 - sig(z(L)))   : z(L) is the non-activated value
        return self.activated_value * (1 - self.activated_value)
    
    def activate(self) : #sigmoid
        self.activated_value = self.sigmoid()
        
    def forward_pass(self, input_vec) :
        self.value = np.dot(self.weights, input_vec) + self.bias
        self.activate()
    
    def cost_derivative_by_layer(self, output_ref) : # 2(y - y_ref)
        return 2*(self.activated_value - output_ref)
    
   
class NeuralNetwork :
    def __init__(self, input_size, hidden_size, output_size, learning_rate) : 
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
    
    
    def backpropagation(self, output_vec, input_vec) :
        hidden_values = np.array([neuron.activated_value for neuron in self.hidden_neurons]) #a(L-1)
        output_errors = []  # store bias_gradients_output
        weight_gradient_output = []
        bias_gradient_hidden = []
        weight_gradient_hidden = []
        
        for i, neuron_out in enumerate(self.output_neurons) : #output backprop
            bias_gradient = neuron_out.sigmoid_derivative() * neuron_out.cost_derivative_by_layer(output_vec[i]) # da(L)/dz(L) * dC0/da(L) ***the last term is a(L-1)***
            output_errors.append(bias_gradient) # we store the bias_gradient of each output neurons
            weight_gradient = hidden_values * bias_gradient # da(L)/dz(L) * dC0/da(L) * dz(L)/dw(L)
            weight_gradient_output.append(weight_gradient) # we store each weights gradients to apply later
        
        for j, neuron_hidden in enumerate(self.hidden_neurons) : # hidden backprop
            sum = 0
            for x, neuron_out in enumerate(self.output_neurons) : # we calculate the propagate sum error fro output to hidden : dC/dz(L-1) = (Σ(dC0/dz(L)) * w(L)) * (da(L-1)/dz(L-1))
                sum += output_errors[x] * neuron_out.weights[j]
            bias_gradient = sum * neuron_hidden.sigmoid_derivative()
            bias_gradient_hidden.append(bias_gradient) # we store them to apply later
            weight_gradient_2 = input_vec * bias_gradient # bias_gradient * dC0/da(L-1)
            weight_gradient_hidden.append(weight_gradient_2) # store also (we technically can apply them now because our h_lay is the last one, but if we put more layer we want to generalize the method)
        
        #weights and biases update
        for i, neuron_out in enumerate(self.output_neurons) : 
            neuron_out.weights -= weight_gradient_output[i] * self.lr 
            neuron_out.bias -= output_errors[i] * self.lr
        for j, neuron_hidden in enumerate(self.hidden_neurons) :
            neuron_hidden.weights -= weight_gradient_hidden[j] * self.lr
            neuron_hidden.bias -= bias_gradient_hidden[j] * self.lr  
            
            
    def train(self, input, output, epoch):
        errors = []
        for i in range(epoch):
            total_error = 0
            for j, sample in enumerate(input):
                self.forward_pass(sample)
                self.backpropagation(output[j], input[j])
                # Calculate total error
                total_error += sum([(neuron.activated_value - output[j][k])**2 for k, neuron in enumerate(self.output_neurons)])
            errors.append(total_error / len(input))
            # Show the MSE every x epoch
            if (i%1000 == 0) :
                print(f"epoch {i}, MSE : {total_error / len(input)}")
        plt.plot(range(epoch), errors, label='Total error')
        plt.xlabel('Epoch')
        plt.ylabel('Mean error')
        plt.title('Error evolution through epoch')
        plt.legend()
        plt.show()
        
    def test(self, input, output_ref) :
        for i, sample in enumerate(input) :
            self.forward_pass(sample)
            output_pred = []
            for neurons in self.output_neurons :
                output_pred.append(round(float(neurons.activated_value), 2)) #we limit the prediction to 2 decimals
            print("predicted : ", output_pred, " | ref : ", output_ref[i])
        

# -----------------------Test of the entire NN-------------------------------     
nn = NeuralNetwork(2, 2, 1, 0.1) #(input_size, hidden_size, output_size, lr)
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = np.array([[0], [1], [1], [0]])
nn.train(input, output, 10000)

nn.test(input, output)


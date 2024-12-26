# ML_XOR_no_library
XOR gate model without using any ML library.  
Library used :
- Numpy
- Matplotlib (local save)

## Modulable Neural Network class
### How to use it ?
1. **Init a neural network with** : `nn = NeuralNetwork(input_size, hidden_size, output_size, lr)`
   - `input_size` : dimension of your input samples (for XOR, it will be 2)
   - `hidden_size` : number of hidden neurons (2 is enough for XOR, but you can add more)
   - `output_size` : number of output neurons (1 for XOR, but you can add more)
   - `lr` : learning rate (I set it to 0.2)

2. **Train your samples with** : `nn.train(input_set, output_set, nb_epoch)`
   - `input_set` : Needs to be a numpy array of samples, each sample is an array of int/float.
     For XOR : `input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`
   - `output_set` : Same as input_set.
     For XOR : `output = np.array([[0], [1], [1], [0]])`
   - `nb_epoch` : The number of epochs/iterations of training through your entire sample set.

3. **Test your network with** : `nn.test(input_set, output_set)`
   - `input_set` : Needs to be a numpy array of samples, each sample is an array of int/float.
     For XOR : `input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`
   - `output_set` : Same as input_set.
     For XOR : `output = np.array([[0], [1], [1], [0]])`

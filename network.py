import numpy as np
import math

class Tensor:

    def __init__ (self, input_size, output_size):

        #Defines the amount of input and output neurons
        self.input_size = input_size
        self.output_size = output_size

        #Defines the variable that'll hold the weights
        self.weights = []

    def initiate_weights (self):

        #Initialising the weights as random values
        for i in xrange(self.output_size):
            self.weights.append([])
            for j in xrange(self.input_size + 1):
                self.weights[i].append(np.random.uniform(-1,1))

    def summation (self, input_):

        #Calculates and returns the output of the neuron layer as an array
        self.output = []
        for i in xrange(self.output_size):
            self.output.append(0)
            for j in xrange(self.input_size + 1):
                if j < self.input_size:
                    self.output[i] += self.weights[i][j] * input_[j]
                else:
                    #Adds the bias
                    self.output[i] += self.weights[i][j] * 1

        return self.output

    @staticmethod
    def activation (function, array):

        #Takes the string and applies the corresponding function
        output = []
        for i in xrange(len(array)):
            if function == "tanh":
                output.append(np.tanh(array[i]))
            elif function == "sigmoid":
                output.append(1/(math.exp(-1 * array[i]) + 1))
            elif function == "relu":
                output.append(max(0,array[i]))
            elif function == "linear":
                output.append(array[i])

        return output

    @staticmethod
    def activation_derivative (function, array):

        #Takes the string and applies the derivative of the corresponding function
        output = []
        for i in xrange(len(array)):
            if function == "tanh":
                output.append(1 - np.tanh(array[i])**2)
            elif function == "sigmoid":
                sigmoid = 1/(math.exp(-1 * array[i]) + 1)
                output.append(sigmoid * (1 - sigmoid))
            elif function == "relu":
                if array[i] > 0:
                    output.append(1)
                else:
                    output.append(0)
            elif function == "linear":
                output.append(1)

        return output

    @staticmethod
    def multi (scalar, matrix, depth):

        #Multiplies the koordinates of a matrix with a scalar
        output = []
        for i in xrange(len(matrix)):
            if depth > 1:
                output.append([])
                for j in xrange(len(matrix[i])):
                    if depth > 2:
                        output[i].append([])
                        for k in xrange(len(matrix[i][j])):
                            output[i][j].append(matrix[i][j][k] * scalar)
                    else:
                        output[i].append(matrix[i][j] * scalar)
            else:
                output.append(matrix[i] * scalar)

        return output

    @staticmethod
    def add (matrix1, matrix2, depth):

        #Adds the coordinates of two matrices
        output = []
        for i in xrange(len(matrix1)):
            if depth > 1:
                output.append([])
                for j in xrange(len(matrix1[i])):
                    if depth > 2:
                        output[i].append([])
                        for k in xrange(len(matrix1[i][j])):
                            output[i][j].append(matrix1[i][j][k] + matrix2[i][j][k])
                    else:
                        output[i].append(matrix1[i][j] + matrix2[i][j])
            else:
                output.append(matrix1[i] + matrix2[i])

        return output

    @staticmethod
    def zero_array (n):

        #Creates a zero vector in n dimensions
        output = []
        for i in xrange(n):
            output.append(0)

        return output

class Network:

    def __init__ (self, input_size, learning_rate):

        #Sets the amount of inputs the network will take
        self.network_size = []
        self.network_size.append(input_size)

        #Defines the learning rate
        self.lr = learning_rate

        #Defines the array that contains the layers
        self.layers = []

    def layer (self, layer_size,function):

        #Adds the layers, specifies the amount of neurons in each and initialises the weights
        for i in xrange(len(layer_size)):
            self.layers.append(Tensor(self.network_size[-1],layer_size[i]))
            self.layers[i].initiate_weights()
            self.network_size.append(layer_size[i])

        #Saving what activation function is used in each layer
        self.function = function

    def feed_forward (self, input_):

        #Feeds forward through the network and saves the output of each neuron

        #After activation
        self.input = [input_]

        #Before activation
        self.layer_output = []

        for i in xrange(len(self.layers)):
            self.sum = self.layers[i].summation(self.input[i])
            self.layer_output.append(self.sum)
            self.input.append(Tensor.activation(self.function[i],self.sum))

        return self.input[-1]

    def backprop (self, loss):

        #Backpropagates through the network to find the gradient for each weight

        #The variable that holds the updates to the weights for the whole network
        self.weight_nabla = []

        #The the partial deriviative of the cost is initiated as the loss
        self.del_cost = loss

        for layer in range(len(self.layers) - 1, -1, -1):

            #The variable that holds the updates to the weights for a specific layer
            self.weight_grad = []

            #The variable that holds the gradient that will be passed on to the previous layer
            self.del_weight = Tensor.zero_array(self.network_size[layer])

            for neuron in xrange(self.network_size[layer + 1]):

                #Finds the derivatives for the activation function and the summation
                self.del_activation = Tensor.activation_derivative(self.function[layer], self.layer_output[layer])
                self.del_layer = self.input[layer]

                #Applying the chain rule to find the gradient for the weights
                self.chain = Tensor.multi(self.del_activation[neuron] * self.del_cost[neuron], self.del_layer,1)

                #Adding the gradient of the bias
                self.chain.append(self.del_activation[neuron] * self.del_cost[neuron])

                #Saving the weight gradient and adding the learning rate
                self.weight_grad.append(Tensor.multi(self.lr, self.chain,1))

                #Applying the chain rule to find the gradient that will be passed on as the cost
                self.chain = Tensor.multi(self.del_activation[neuron] * self.del_cost[neuron], self.layers[layer].weights[neuron][:-1],1)
                self.del_weight = Tensor.add(self.chain, self.del_weight, 1)

            #Saving the weight gradients for the whole layer
            self.weight_nabla.insert(0, self.weight_grad)

            #The cost is updated so the next layer gets the correct gradient
            self.del_cost = self.del_weight

    def update_weights (self, gradients):

        #Updating the weights for the whole network with the weight gradient
        for i in xrange(len(self.layers)):
            self.layers[i].weights = Tensor.add(self.layers[i].weights, Tensor.multi(-1, gradients[i], 2), 2)

    def train (self, input_, mini_batch, training_size):

        self.training_size = training_size

        for i in xrange((len(input_)*self.training_size)/100):
            self.input = input_[i][0]

            #The gradient of the loss is calculated
            self.guess = self.feed_forward(self.input)
            self.loss =  Tensor.add(self.guess, Tensor.multi(-1, input_[i][1], 1), 1)

            #The backpropagation is done
            self.backprop(self.loss)

            #The average gradient for each batch is calculated
            if i > 0 and self.updated == False:
                self.weight_update = Tensor.add(self.weight_update, self.weight_nabla, 3)

            else:
                self.weight_update = self.weight_nabla
                self.updated = False

            #The weights are updated
            if (i + 1) % mini_batch == 0:
                Tensor.multi(1/mini_batch,self.weight_update,3)
                self.update_weights(self.weight_update)
                self.updated = True


    def test (self, input_, loss_type):

        #The loss is calculated and returned
        self.error = 0
        self.length = 0
        if loss_type == "test":
            self.test_points = range((len(input_)*self.training_size)/100,len(input_))
            self.length = len(input_) - (len(input_)*self.training_size)/100
        elif loss_type == "training":
            self.test_points = xrange((len(input_)*self.training_size)/100)
            self.length = (len(input_)*self.training_size)/100

        for i in self.test_points:
            self.input = input_[i][0]
            self.feed = self.feed_forward(self.input)
            for j in xrange(len(input_[i][0])):
                self.error += ((self.feed[j] - input_[i][1][j])**2)/self.length

        return self.error

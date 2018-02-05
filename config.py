#Change these parameters to change the network

#Size of the dataset
Datasize = 300

#The learning rate
Learning_rate = 0.01

#Structure of the hidden layers. The input and output are hardcoded to two neurons
Network_structure = [5,3]

#Iterations per update
Batch_size = 10

#Epochs per frame
Epochs = 1

#In percent
Training_size = 90

#Choose between cross, circle and linear
Data_type = "cross"

#Choose between tanh, sigmoid, relu and linear
Activation_function = "tanh"

#Initialising values
Epoch = 0
reset = True
firstTime = True
train = False

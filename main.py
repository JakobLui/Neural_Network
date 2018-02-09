import numpy as np
import pygame, sys
from dataset import *
from network import *
from config import *
from interface import *

pygame.init()

def create_window():
    global window, window_height, window_width, window_title
    window_width, window_height = screen_width, screen_height
    window_title = "Neural Network"
    pygame.display.set_caption(window_title)
    window = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE|pygame.DOUBLEBUF)

#Creating the window and defining variables needed for visualising the network
create_window()

#Initialising values
isRunning = True
Epoch = 0
reset = True
firstTime = True
train = False

#The loop that visualises, trains and runs the network
while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False

    #Resets the network and dataset
    if reset == True:

        #Sets the start value of the epock to 0
        Epoch = 0

        #A dataset is generated
        x = Dataset(2,Datasize,Data_type)

        #The network and the learning rate is defined
        net = Network(2, Learning_rate)

        #The network structure and the activation functions are specified and the weights are intialised
        activ_func = []
        Network_structure.append(2)
        for i in xrange(len(Network_structure)):
            activ_func.append(Activation_function)
        net.layer(Network_structure,activ_func)

        reset = False

    #Passes each pixel of the visualisation through network to determine the class and then draws the pixel

    network_prediction(window, net, firstTime)

    #Trains the network and adds one to the epoch counter
    if train == True:
        for i in xrange(Epochs):
            net.train(x.data, Batch_size, Training_size)
            Epoch = Epoch + 1

    #Draws each point of the dataset and colors it according the the class
    show_dataset(window, x)

    #Shows the test and training loss for the network while it's running
    draw_loss(window, window_height, window_width, net, x, train)

    #Shows the amount of epochs the network has done
    draw_epoch(window, window_height, window_width, Epoch)

    mouse = pygame.mouse.get_pos()
    #Draws the stop button
    stop_button(window,window_height + 115, window_height - 65, mouse, train)

    #Draws the play/pause button
    play_button(window,window_height + 45, window_height - 65, mouse, train)

    clicked = False
    #Checks for click event and checks which button has been pressed
    for event in pygame.event.get():

        if event.type == pygame.MOUSEBUTTONUP:
            clicked = True

        if clicked and np.sqrt((window_height + 115 - mouse[0])**2 + (window_height - 65 - mouse[1])**2) < 25:
            isRunning = False
        elif clicked and np.sqrt((window_height + 45 - mouse[0])**2 + (window_height - 65 - mouse[1])**2) < 25:
            if train:
                train = False
                reset = True
            else:
                train = True

    if firstTime:
        firstTime = False

    pygame.display.update()

pygame.quit()
sys.exit()

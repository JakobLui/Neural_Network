import pygame, sys
from threading import *
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

def Main():

    #Initialising values
    isRunning = True
    Epoch = 0
    reset = True
    firstTime = True
    train = False

    #Creating the window and defining variables needed for visualising the network
    create_window()

    #The loop that visualises, trains and runs the network
    while isRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRunning = False
        clicked = False

        mouse = pygame.mouse.get_pos()

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

            net.threaded_train([[[0,0],[0,0]]], 1, Training_size)

            reset = False

        #Passes each pixel of the visualisation through network to determine the class and then draws the pixel
        thread = Thread (target = network_prediction, args = (window, net, firstTime))
        thread.start()
        thread.join()

        #Trains the network and adds one to the epoch counter
        if train == True:
            for i in xrange(Epochs):
                if net.thread.isAlive() == False:
                    net.threaded_train(x.data, Batch_size, Training_size)
                    Epoch = Epoch + 1


        #Draws each point of the dataset and colors it according the the class
        show_dataset(window, x)

        #Shows the test and training loss and the eochs for the network while it's running
        draw_loss(window, window_height, window_width, net, x, train, Epoch)

        #Draws the stop button
        stop_button(window,window_height + 115, window_height - 65, mouse, train)

        #Draws the play/pause button
        play_button(window,window_height + 45, window_height - 65, mouse, train)

        #Checks for click event and checks which button has been pressed
        for event in pygame.event.get():

            if event.type == pygame.MOUSEBUTTONUP:
                clicked = True

            if clicked and ((window_height + 115 - mouse[0])**2 + (window_height - 65 - mouse[1])**2)**0.5 < 25:
                isRunning = False
            elif clicked and ((window_height + 45 - mouse[0])**2 + (window_height - 65 - mouse[1])**2)**0.5 < 25:
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

if __name__ == "__main__":
    Main()

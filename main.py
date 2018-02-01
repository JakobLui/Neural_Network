import numpy as np
import pygame, sys
from dataset import *
from network import *

#Change these parameters to change the network 
Datasize = 300
Learning_rate = 0.01
Network_structure = [5,3]
Batch_size = 10
Epochs = 1
Training_size = 90
Data_type = "cross"
Activation_function = "tanh"



pygame.init()

pygame.font.init()

def create_window():
    global window, window_height, window_width, window_title
    window_width, window_height = 560, 400
    window_title = "Neural Network"
    pygame.display.set_caption(window_title)
    window = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE|pygame.DOUBLEBUF)

#Creating the window and defining variables needed for visualising the network
create_window()

scale = 10
row = window_height / scale
col = window_height / scale

orange = (255,125,20)
blue = (100,100,255)
node_x = xrange(col)
node_y = xrange(row)
color = orange
firstTime = True
classification = []
Epoch = 0
reset = True
train = False

myfont = pygame.font.SysFont("Arial", 30)

isRunning = True

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
    for i in node_x:
        for j in node_y:
            net_node_x = ((float(i + 0.5)/float(col))*2) - 1
            net_node_y = ((float(j + 0.5)/float(row))*2) - 1
            feed = net.feed_forward([net_node_x, net_node_y])
            if feed[0] < feed[1]:
                if feed[1] == 0:
                    feed[1] = 0.00001
                R = min(255, 100 + (100 * abs(feed[0] / feed[1])))
                G = min(255, 100 + (100 * abs(feed[0] / feed[1])))
                color = (R,G,255)
            else:
                if feed[0] == 0:
                    feed[0] = 0.00001
                G = min(255,125 + (100 * abs(feed[1] / feed[0])))
                B = min(255,20 + (100 * abs(feed[1] / feed[0])))
                color = (255,G,B)

            #Checks if the pixel has changed and draws it if it has
            if firstTime:
                classification.append(color)
                pygame.draw.rect(window, color, (i * scale, j * scale, scale, scale), 0)
            elif color != classification[i * col + j]:
                classification[i * col + j] = color
                pygame.draw.rect(window, color, (i * scale, j * scale, scale, scale), 0)

    #Draws each point of the dataset and colors it according the the class
    for i in xrange((len(x.data)*Training_size)/100):
        if x.data[i][1][0] < x.data[i][1][1]:
            color = blue
        else:
            color = orange

        point_x = int(round((x.data[i][0][0] + 1) * col * scale/2))
        point_y = int(round((x.data[i][0][1] + 1) * row * scale/2))
        pygame.draw.circle(window, color, (point_x, point_y), 5, 0)
        pygame.draw.circle(window, (0,0,0), (point_x, point_y), 5, 1)

    #Trains the network and adds one the the epoch counter
    if train == True:
        for i in xrange(Epochs):
            net.train(x.data, Batch_size, Training_size)
            Epoch = Epoch + 1

    if firstTime:
        firstTime = False

    #Shows the test and training loss for the network while it's running
    if train:
        Test_loss = myfont.render("{}".format(round(net.test(x.data,"test"),10)), False, (0,0,0))
        Training_loss = myfont.render("{}".format(round(net.test(x.data,"training"),10)), False, (0,0,0))
    else:
        Test_loss = myfont.render("0", False, (0,0,0))
        Training_loss = myfont.render("0", False, (0,0,0))

    pygame.draw.rect(window, (200,200,200), (window_height,0,window_width - window_height,window_height), 0)

    test_loss = myfont.render("Test Loss:", False, (0,0,0))
    training_loss = myfont.render("Training Loss:", False, (0,0,0))
    epoch = myfont.render("Epoch: {}".format(Epoch), False, (0,0,0))

    window.blit(training_loss,(window_width - 150, 130))
    window.blit(Training_loss,(window_width - 150, 155))
    window.blit(test_loss,(window_width - 150, 55))
    window.blit(Test_loss,(window_width - 150, 80))
    window.blit(epoch,(window_width - 150, 5))

    clicked = False
    mouse = pygame.mouse.get_pos()

    #Draws the stop button
    if np.sqrt((window_height + 125 - mouse[0])**2 + (window_height - 65 - mouse[1])**2) < 25:
        pygame.draw.circle(window, (120,120,120), (window_height + 115, window_height - 65),25,0)
        pygame.draw.circle(window, (50,50,50), (window_height + 115, window_height - 65),25,2)
        pygame.draw.rect(window,(100,100,100),(window_height + 105, window_height - 75,20,20),0)

    else:
        pygame.draw.circle(window, (170,170,170), (window_height + 115, window_height - 65),25, 0)
        pygame.draw.circle(window, (100,100,100), (window_height + 115, window_height - 65),25,2)
        pygame.draw.rect(window,(150,150,150),(window_height + 105, window_height - 75,20,20),0)

    #Draws the play/pause button
    if np.sqrt((window_height + 45 - mouse[0])**2 + (window_height - 65 - mouse[1])**2) < 25:
        pygame.draw.circle(window, (120,120,120), (window_height + 45, window_height - 65),25,0)
        pygame.draw.circle(window, (50,50,50), (window_height + 45, window_height - 65),25,2)
        if train == False:
            pygame.draw.polygon(window, (100,100,100),((window_height + 35, window_height - 55), (window_height + 35, window_height - 75), (window_width - 102, window_height - 65)), 0)
        else:
            pygame.draw.rect(window,(100,100,100),(window_height + 35, window_height - 75,7,20),0)
            pygame.draw.rect(window,(100,100,100),(window_height + 48, window_height - 75,7,20),0)
    else:
        pygame.draw.circle(window, (170,170,170), (window_height + 45, window_height - 65),25, 0)
        pygame.draw.circle(window, (100,100,100), (window_height + 45, window_height - 65),25,2)
        if train == False:
            pygame.draw.polygon(window, (150,150,150),((window_height + 35, window_height - 55), (window_height + 35, window_height - 75), (window_width - 102, window_height - 65)), 0)
        else:
            pygame.draw.rect(window,(150,150,150),(window_height + 35, window_height - 75,7,20),0)
            pygame.draw.rect(window,(150,150,150),(window_height + 48, window_height - 75,7,20),0)

    #Checks for click event and checks which button has been pressed
    for event in pygame.event.get():

        if event.type == pygame.MOUSEBUTTONUP:
            clicked = True

        if clicked and np.sqrt((window_height + 125 - mouse[0])**2 + (window_height - 65 - mouse[1])**2) < 25:
            isRunning = False
        elif clicked and np.sqrt((window_height + 45 - mouse[0])**2 + (window_height - 65 - mouse[1])**2) < 25:
            if train:
                train = False
                reset = True
            else:
                train = True

    pygame.display.update()

pygame.quit()
sys.exit()

import numpy as np
import pygame, sys
from config import Training_size

pygame.font.init()

scale = 10
classification = []
orange = (255,125,20)
blue = (100,100,255)
color = orange
screen_width, screen_height = 560, 400
row = screen_height / scale
col = screen_height / scale
node_x = xrange(col)
node_y = xrange(row)
myfont = pygame.font.SysFont("Arial", 30)

def stop_button (screen, x, y, mouse, train):
    #Draws the stop button
    #x = screen_height + 115
    #y = screen_height - 65
    if np.sqrt((x - mouse[0])**2 + (y - mouse[1])**2) < 25:
        pygame.draw.circle(screen, (120,120,120), (x, y),25,0)
        pygame.draw.circle(screen, (50,50,50), (x, y),25,2)
        pygame.draw.rect(screen,(100,100,100),(x - 10, y - 10,20,20),0)

    else:
        pygame.draw.circle(screen, (170,170,170), (x, y),25, 0)
        pygame.draw.circle(screen, (100,100,100), (x, y),25,2)
        pygame.draw.rect(screen,(150,150,150),(x - 10, y - 10,20,20),0)
    pass

def play_button (screen, x, y, mouse, train):
    #Draws the play/pause button
    if np.sqrt((x - mouse[0])**2 + (y - mouse[1])**2) < 25:
        pygame.draw.circle(screen, (120,120,120), (x, y),25,0)
        pygame.draw.circle(screen, (50,50,50), (x, y),25,2)
        if train == False:
            pygame.draw.polygon(screen, (100,100,100),((x - 10, y + 10), (x - 10, y - 10), (x + 13, y)), 0)
        else:
            pygame.draw.rect(screen,(100,100,100),(x - 10, y - 10,7,20),0)
            pygame.draw.rect(screen,(100,100,100),(x + 3, y - 10,7,20),0)
    else:
        pygame.draw.circle(screen, (170,170,170), (x, y),25, 0)
        pygame.draw.circle(screen, (100,100,100), (x, y),25,2)
        if train == False:
            pygame.draw.polygon(screen, (150,150,150),((x - 10, y + 10), (x - 10, y - 10), (x + 13, y)), 0)
        else:
            pygame.draw.rect(screen,(150,150,150),(x - 10, y - 10,7,20),0)
            pygame.draw.rect(screen,(150,150,150),(x + 3, y - 10,7,20),0)

def draw_loss (screen, x, y, network, dataset, train):
    if train:
        Test_loss = myfont.render("{}".format(round(network.test(dataset.data,"test"),10)), False, (0,0,0))
        Training_loss = myfont.render("{}".format(round(network.test(dataset.data,"training"),10)), False, (0,0,0))
    else:
        Test_loss = myfont.render("0", False, (0,0,0))
        Training_loss = myfont.render("0", False, (0,0,0))

    pygame.draw.rect(screen, (200,200,200), (x,0,y - x,x), 0)

    test_loss = myfont.render("Test Loss:", False, (0,0,0))
    training_loss = myfont.render("Training Loss:", False, (0,0,0))

    screen.blit(training_loss,(y - 150, 130))
    screen.blit(Training_loss,(y - 150, 155))
    screen.blit(test_loss,(y - 150, 55))
    screen.blit(Test_loss,(y - 150, 80))

def draw_epoch (screen, x, y, Epoch):
    epoch = myfont.render("Epoch: {}".format(Epoch), False, (0,0,0))
    screen.blit(epoch,(y - 150, 5))

def show_dataset (screen, dataset):
    for i in xrange((len(dataset.data)*Training_size)/100):
        if dataset.data[i][1][0] < dataset.data[i][1][1]:
            color = (100,100,255)
        else:
            color = (255,125,20)

        point_x = int(round((dataset.data[i][0][0] + 1) * col * scale/2))
        point_y = int(round((dataset.data[i][0][1] + 1) * row * scale/2))
        pygame.draw.circle(screen, color, (point_x, point_y), 5, 0)
        pygame.draw.circle(screen, (0,0,0), (point_x, point_y), 5, 1)

def network_prediction (screen, network, firstTime):
    #Passes each pixel of the visualisation through network to determine the class and then draws the pixel
    for i in node_x:
        for j in node_y:
            net_node_x = ((float(i + 0.5)/float(col))*2) - 1
            net_node_y = ((float(j + 0.5)/float(row))*2) - 1
            feed = network.feed_forward([net_node_x, net_node_y])
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
                pygame.draw.rect(screen, color, (i * scale, j * scale, scale, scale), 0)
            elif color != classification[i * col + j]:
                classification[i * col + j] = color
                pygame.draw.rect(screen, color, (i * scale, j * scale, scale, scale), 0)

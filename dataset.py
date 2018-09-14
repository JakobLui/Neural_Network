import numpy as np

class Dataset:

    #Generates a random dataset with classes
    @staticmethod
    def dist (array):

        output = 0
        for i in xrange(len(array)):
             output += array[i]**2

        return np.sqrt(output)

    def __init__ (self, features, size, data_type, feature):
        #Creates the dataset
        self.features = features
        self.size = size
        self.data_type = data_type
        self.data = []
        for i in xrange(size):
            self.data.append([])
            self.point = []
            for j in xrange(features):
                self.point.append(np.random.uniform(-1,1))
            self.data[i].append(self.point)


        #Specifies the class for each point

        if data_type == "circle":
            for i in xrange(size):
                if Dataset.dist(self.data[i][0]) < 0.75:
                    self.data[i].append([0,1])
                else:
                    self.data[i].append([1,0])

        elif data_type == "cross":
            for i in xrange(size):
                if self.data[i][0][0] > 0 and self.data[i][0][1] > 0:
                    self.data[i].append([0,1])
                elif self.data[i][0][0] < 0 and self.data[i][0][1] < 0:
                    self.data[i].append([0,1])
                else:
                    self.data[i].append([1,0])

        elif data_type == "linear":
            for i in xrange(size):
                if self.data[i][0][0] > self.data[i][0][1]:
                    self.data[i].append([0,1])
                else:
                    self.data[i].append([1,0])

        #Checks the feature and changes accordingly
        if feature == "squared":
            for i in xrange(size):
                for j in xrange(2):
                    self.data[i][0][j] = self.data[i][0][j]**2

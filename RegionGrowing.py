import numpy as np

class RegionGrowing:
    def __init__(self, grayScaleImg, threshold):
        self.image = np.array(grayScaleImg, copy = True)
        self.labels = 0
        self.labelImg = np.full(grayScaleImg.shape, -1, dtype = int)
        self.threshold = threshold

    def addRegion(self, seedPoint):
        label = self.labels

        if self.labelImg[seedPoint[0], seedPoint[1]] == -1:
            self.labelImg[seedPoint[0], seedPoint[1]] = label
        else:
            print "This region is already added."
            return self.labelImg[seedPoint[0], seedPoint[1]]

        neighbors = self.getFourNeighbors(seedPoint)
        regionMeanValue = int(self.image[seedPoint[0], seedPoint[1]])

        while neighbors:
            row, col = neighbors.pop()
            if np.abs(regionMeanValue - self.image[row, col]) < self.threshold:
                self.labelImg[row, col] = label
                neighbors.extend(self.getFourNeighbors((row, col)))
                regionMeanValue = np.mean(self.image[self.labelImg == label])

        self.image[self.labelImg == label] = regionMeanValue
        print "A new region added successfully."
        self.labels += 1
        return label

    def getFourNeighbors(self, seedPoint):
        delta = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        neighbors = []
        for d in delta:
            neighbor = np.asarray(seedPoint) + d
            try:
                if self.labelImg[neighbor[0], neighbor[1]] == -1 : neighbors.append(neighbor)
            except:
                continue
        return neighbors

    def getEightNeighbors(self, seedPoint):
        delta = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        neighbors = []
        for d in delta:
            neighbor = tuple(sum(p) for p in zip(d, seedPoint))
            try:
                if self.labelImg[neighbor[0], neighbor[1]] == -1 : neighbors.append(neighbor)
            except:
                continue
        return neighbors

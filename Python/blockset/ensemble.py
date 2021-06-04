import pyblockset

class BlocksetBase():
    def __init__(self):
        self.model = pyblockset.BlocksetBase()

    def initRandomForestClassifier(self):
        self.model.initRandomForestClassifier()

    def initGradientBoostedClassifier(self):
        self.model.initGradientBoostedClassifier()

    def initRandomForestRegressor(self):
        self.model.initRandomForestRegressor()

    def initGradientBoostedRegressor(self):
        self.model.initGradientBoostedRegressor()

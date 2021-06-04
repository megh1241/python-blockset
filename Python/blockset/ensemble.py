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

    def loadJSONModel(self, filename):
        self.model.loadJSONModel(filename)
    
    def loadBlocksetModel(self, filename):
        self.model.loadBlocksetModel(filename)
    
    def pack(self, filename):
        self.model.pack(filename)
    
    def pack(self):
        self.model.pack()
    
    def predict(X):
        return self.model.predict(X)

import pyblockset

class BlocksetBase():
    def __init__(self):
        self.model = pyblockset.BlocksetBase()
        self.task='c'

    def initRandomForestClassifier(self):
        self.model.initRandomForestClassifier()

    def initGradientBoostedClassifier(self):
        self.model.initGradientBoostedClassifier()

    def initRandomForestRegressor(self):
        self.model.initRandomForestRegressor()
        self.task='r'

    def initGradientBoostedRegressor(self):
        self.model.initGradientBoostedRegressor()
        self.task='r'

    def loadJSONModel(self, filename):
        self.model.loadJSONModel(filename)
    
    def loadBlocksetModel(self, filename):
        self.model.loadBlocksetModel(filename)
    
    def pack(self, filename):
        self.model.pack(filename)
    
    def pack(self):
        self.model.pack()
    
    def serialize(self, filename):
        self.model.serialize(filename)
    
    def predict(self, X):
        if self.task == 'r':
            return self.model.predictLabelRegression(X)
        else:
            return self.model.predictLabelClassification(X)

import pyblockset

class BlocksetBase():
    def __init__(self):
        self.model = pyblockset.BlocksetBase()
        self.task='c'
        self.algorithm='rf'

    def initRandomForestClassifier(self):
        self.model.initRandomForestClassifier()

    def initGradientBoostingClassifier(self):
        self.model.initGradientBoostedClassifier()
        self.algorithm='gb'

    def initRandomForestRegressor(self):
        self.model.initRandomForestRegressor()
        self.task='r'

    def initGradientBoostingRegressor(self):
        self.model.initGradientBoostedRegressor()
        self.task='r'
        self.algorithm='gb'

    def loadJSONModel(self, filename):
        if self.task == 'r' and self.algorithm=='gb':
            init_model_filename = 'init' + filename[:-5] + '.joblib'
            pyblockset.Config.setConfigItem('initModelFilename', init_model_filename)
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
            if self.algorithm == 'gb':
                print("enter gb regression!!!!!!!!")
                import joblib
                init_model_filename = pyblockset.Config.getValue('initModelFilename')
                init_model = joblib.load(init_model_filename)
                A = X.reshape(1, -1)
                print(init_model.predict(A)[0])
                return self.model.predictLabelRegression(X) + init_model.predict(A)[0]

            return self.model.predictLabelRegression(X)
        else:
            return self.model.predictLabelClassification(X)

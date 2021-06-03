import pyblockset

class BlocksetBase():
    def __init__(self):
        self.model = pyblockset.BlocksetBase()

    def initRandomForestClassifier(self):
        self.model.initRandomForestClassifier()


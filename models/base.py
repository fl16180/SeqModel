

class BaseModel:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

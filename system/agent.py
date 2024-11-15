import importlib

class Agent:
    def __init__(self, algorithm, model, actions):
        m = importlib.import_module(algorithm)
        self.algorithm = getattr(m,"Algorithm")(model, actions)

    def stop(self):
        return self.algorithm.stop()

    def train(self, data):
        self.algorithm.train(data)

    def step(self, data, custom):
        if custom != None:
            return custom
        return self.algorithm.step(data)

    def last(self, data, done):
        return self.algorithm.last(data, done)

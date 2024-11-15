import yaml, pickle

def load(path, mode, fn):
    with open(path, mode) as file:
        return fn(file)

def save(path, mode, fn, save):
    with open(path, mode) as file:
        fn(save, file)

def load_yaml(path):
    load(path, "r", yaml.safe_load)

def load_pickle(path):
    return load(path, "rb", pickle.load)

def save_pickle(path, save):
    with open(path, "wb") as file:
        pickle.dump(save, file)

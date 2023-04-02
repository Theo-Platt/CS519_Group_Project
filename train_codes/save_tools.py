import pickle

def save_pickle(path, model):
    pickle.dump(model, open(path, 'wb'))
    

def load_pickle(path):
    loaded_model = pickle.load(open(path, 'rb'))
    return loaded_model

def save_raw(path, params):
    pass

def load_raw(path):
    pass
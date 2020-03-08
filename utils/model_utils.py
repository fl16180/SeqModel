import pickle
from constants import PROCESSED_DIR


def save_model(model, project, name):
    fname = PROCESSED_DIR / project / 'models' / f'saved_model_{name}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(model, f)


def load_model(project, name):
    fname = PROCESSED_DIR / project / 'models' / f'saved_model_{name}.pkl'
    with open(fname, 'rb') as f:
        model = pickle.load(f)
    return model

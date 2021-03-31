from dill import dill


def dump(o, filename):
    with open(filename, 'wb') as f:
        dill.dump(o, f)


def load(filename):
    with open(filename, 'rb') as f:
        dill.load(f)


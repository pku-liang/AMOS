class Model:

    def __init__(self): pass

    def __call__(self, inputs):
        raise NotImplementedError

    @property
    def weights(self):
        raise NotImplementedError

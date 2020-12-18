class Module:

    def __init__(self, name):
        self.name = name

    def __call__(self, *inputs):
        raise NotImplementedError

    @property
    def weights(self):
        raise NotImplementedError

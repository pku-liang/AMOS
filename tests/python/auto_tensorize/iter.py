class myiterable(object):
    def __init__(self, x):
        self.x = x

    def __iter__(self):
        return iter(self.x)



if __name__ == "__main__":
    for i in myiterable(range(10)):
        print(i)
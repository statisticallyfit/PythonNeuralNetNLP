class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for key, val in kwargs.items():
            setattr(self, key, val)


    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

    def update(self, fromDict: dict):
        for key, val in fromDict.items():
            self.set(key, val)

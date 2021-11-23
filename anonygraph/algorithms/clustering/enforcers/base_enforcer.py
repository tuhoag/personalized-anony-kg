class BaseEnforcer:
    def __init__(self, name, args):
        self.__name = name
        self.__args = args

    @property
    def name(self):
        return self.__name

    def __call__(self):
        """
        docstring
        """
        pass

    def call(self):
        """
        docstring
        """
        pass
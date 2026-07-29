from enum import Enum


class Token_Type(Enum):
    BEAR = 1
    FOX = 2
    HAWK = 3
    DEER = 4
    SALMON = 5


class Token():

    def __init__(self, type: Token_Type, x: int, y: int, width: int) -> None:
        self.type = type
        self.x = x
        self.y = y
        self.width = width
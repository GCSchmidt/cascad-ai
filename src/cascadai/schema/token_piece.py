from enum import Enum


class Token_Type(Enum):
    BEAR = 0
    DEER = 1
    SALMON = 2
    HAWK = 3
    FOX = 4


class Token():

    def __init__(self, type: Token_Type, x: int, y: int, width: int) -> None:
        self.type = type
        self.x = x
        self.y = y
        self.width = width
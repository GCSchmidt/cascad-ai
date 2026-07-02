class TilePiece:

    def __init__(self, x: float = 0, y: float = 0, theta: float = 0, side_length: float = 1) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.side_length = side_length
        self.environment_1 = None
        self.environment_2 = None

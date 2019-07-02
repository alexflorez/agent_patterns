import numpy as np
import random
from itertools import product

class Cell:
    def __init__(self):
        self.state = random.randint(0, 1)

    def __repr__(self):
        class_name = type(self).__name__
        return '{}<{!r}>'.format(class_name, self.state)


class CA:
    generation = 0

    def __init__(self, x, y):
        self.rows = x
        self.columns = y
        self.board = self.create_board(x, y)

    def create_board(self, x, y):
        """ start values between 0 and 1 """
        # self.board = np.random.randint(2, size=(self.rows, self.columns))
        board = np.empty((x, y), dtype=object)
        for i, j in product(range(x), range(y)):
            board[i][j] = Cell()
        return board

    def ruleset(self, current, neighbors):
        next_state = 0
        if current == 1 and neighbors <  2:
            next_state = 0           # Loneliness
        elif current == 1 and neighbors >  3:
            next_state = 0           # Overpopulation
        elif current == 0 and neighbors == 3:
            next_state = 1           # Reproduction
        else:
            next_state = current  # Stasis
        return next_state

    def num_neighbors(self, current, x, y):
        """ Add up all the states in a 3x3 surrounding grid """
        rows, columns = self.rows, self.columns
        neighbors = 0
        for i, j in product(range(-1, 2), range(-1, 2)):
            ix = (x + i + rows) % rows
            jy = (y + j + columns) % columns
            neighbors += self.board[ix][jy].state
    
        # Subtract the current cell's state since
        # we added it in the above loop
        neighbors -= current
        return neighbors

    def generate(self):
        """ Process to create a new generation """
        rows, columns = self.rows, self.columns
        next_board = self.create_board(rows, columns)
        for x, y in product(range(rows), range(columns)):
            neighbors = self.num_neighbors(self.board[x][y].state, x, y)
            # Rules of Life
            next_board[x][y].state = self.ruleset(self.board[x][y].state, neighbors)

        self.generation += 1
        self.board = np.copy(next_board)

    def __repr__(self):
        class_name = type(self).__name__
        return '{}<{!r}>'.format(class_name, self.board)

if __name__ == '__main__':
    w = CA(4, 5)
    print(w)
    # w.generate()

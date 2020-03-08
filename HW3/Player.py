import numpy as np
import random
import math
from datetime import datetime

# credit to:
# https://github.com/AchintyaAshok/Connect4-AI-Final-Project
# https://github.com/caiespin/Connect4_Alpha-beta_Expectimax_Search_AI
# https://github.com/KeithGalli/Connect4-Python


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def find_valid_columns(self, board):
        valid_cols = []
        for col in range(board.shape[1]):
            if board[0, col] == 0:
                valid_cols.append(col)
        return valid_cols

    def find_row_to_drop(self, board, col):
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                return r

    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def check_win(self, board, piece):
        # Check horizontal locations for win
        for c in range(7 - 3):
            for r in range(5):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                    c + 3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(7):
            for r in range(6 - 3):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                    c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(7 - 3):
            for r in range(6 - 3):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and \
                        board[r + 3][c + 3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(7 - 3):
            for r in range(3, 5):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and \
                        board[r - 3][c + 3] == piece:
                    return True

    def game_completed(self, board):
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)
        return self.check_win(board, 1) or self.check_win(board, 2) or len(valid_cols) == 0

    def sliding_window(self, window, piece):
        score = 0
        opp_piece = 1
        if piece == 1:
            opp_piece = 2

        count_piece = window.count(piece)
        count_other_piece = window.count(opp_piece)
        count_zero = window.count(0)
        if count_piece == 4 and count_zero == 0:
            score += 10000000
        elif count_piece == 3 and count_zero == 1:
            score += 50
        elif count_piece == 2 and count_zero == 2:
            score += 20
        elif count_piece == 1 and count_zero == 3:
            score += 5

        if count_other_piece == 4 and count_zero == 0:
            score -= 10000000
        elif count_other_piece == 3 and count_zero == 1:
            score -= 50
        elif count_other_piece == 2 and count_zero == 2:
            score -= 20
        elif count_other_piece == 1 and count_zero == 3:
            score -= 5

        return score

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        is_terminal = self.game_completed(board)
        if is_terminal:
            if self.check_win(board, self.player_number):
                return math.inf
            elif self.check_win(board, 3 - self.player_number):
                return -math.inf
            else:
                return 0

        score = 0

        # center col
        center_array = [int(i) for i in list(board[:, 3])]
        center_count = center_array.count(self.player_number)
        score += center_count * 30

        # horizontal
        for r in range(6):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(7 - 3):
                window = row_array[c:c + 4]
                score += self.sliding_window(window, self.player_number)

        # vertical
        for c in range(7):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(6 - 3):
                window = col_array[r:r + 4]
                score += self.sliding_window(window, self.player_number)

        # diagonal
        for r in range(6 - 3):
            for c in range(7 - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.sliding_window(window, self.player_number)

        for r in range(6 - 3):
            for c in range(7 - 3):
                window = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.sliding_window(window, self.player_number)

        return score

    def minimax(self, board, piece, depth, alpha, beta, maximizingPlayer):
        opp_piece = 1
        if piece == 1:
            opp_piece = 2
        valid_locations = self.find_valid_columns(board)
        is_terminal = self.game_completed(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_win(board, self.player_number):
                    return None, math.inf
                elif self.check_win(board, 3 - self.player_number):
                    return None, -math.inf
                else:  # Game is over, no more valid moves
                    return None, 0
            else:  # Depth is zero
                return None, self.evaluation_function(board)
        if maximizingPlayer:
            value = -math.inf
            column = valid_locations[0]
            for col in valid_locations:
                row = self.find_row_to_drop(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, piece)
                new_score = self.minimax(b_copy, opp_piece, depth - 1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else:  # Minimizing player
            value = math.inf
            column = valid_locations[0]
            for col in valid_locations:
                row = self.find_row_to_drop(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, piece)
                new_score = self.minimax(b_copy, opp_piece, depth - 1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def expecti_minimax(self, board, piece, depth, maximizingPlayer):
        other_piece = 1
        if piece == 1:
            other_piece = 2
        valid_locations = self.find_valid_columns(board)
        is_terminal = self.game_completed(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_win(board, self.player_number):
                    return None, math.inf
                elif self.check_win(board, 3 - self.player_number):
                    return None, -math.inf
                else:
                    return None, 0
            else:
                return None, self.evaluation_function(board)

        if maximizingPlayer:  # maximizing
            value = -math.inf
            column = valid_locations[0]
            for col in valid_locations:
                row = self.find_row_to_drop(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, piece)
                new_score = self.expecti_minimax(b_copy, other_piece, depth - 1, False)[1]
                if new_score > value:
                    value = new_score
                    column = col

        else:  # chance node
            value = 0
            column = valid_locations[0]
            for col in valid_locations:
                row = self.find_row_to_drop(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, piece)
                value += self.expecti_minimax(b_copy, other_piece, depth - 1, True)[1] / 7
        return column, value

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        # before = datetime.now()
        piece = self.player_number
        col, minimax_score = self.minimax(board, piece, 4, -math.inf, math.inf, True)
        # after = datetime.now()
        # print("alpha-beta time: {0}".format(after - before))
        return col

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        # before = datetime.now()
        piece = self.player_number
        col, score = self.expecti_minimax(board, piece, 4, True)
        # after = datetime.now()
        # print("alpha-beta time: {0}".format(after - before))
        return col


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

# main function is for debugging
# if __name__ == '__main__':
#     board = np.array([[1, 0, 0, 0, 0, 0, 0],
#                       [1, 0, 0, 0, 0, 0, 0],
#                       [2, 0, 0, 2, 0, 0, 0],
#                       [1, 0, 0, 1, 0, 0, 0],
#                       [1, 1, 1, 1, 0, 0, 0],
#                       [1, 2, 2, 1, 2, 2, 2]])
#
#     ai_player = AIPlayer(1)
#     eval = ai_player.evaluation_function(board)
#     print(eval)
#
#     now = datetime.now()
#     print("time: {0}".format(now))
#
#     col = ai_player.get_expectimax_move(board)
#     print("col: {0}".format(col))
#
#     now = datetime.now()
#     print("time: {0}".format(now))

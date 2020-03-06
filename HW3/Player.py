import numpy as np
import copy
import math
import random


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.opponent_number = 2 if player_number == 1 else 1
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    '''find possible children, return cols that can put pieces'''
    def get_possible_moves(self, board):
        # priority in middle
        cols = np.array([0, 1, 2, 3, 4, 5, 6])
        return cols[board[0] == 0]

    '''drop piece to the board'''
    def drop_piece(self, board, column, piece):
        for row in range(5, -1, -1):
            if board[row][column] == 0:
                board[row][column] = piece
                return
        assert False

    '''to check if the game complete'''
    def game_completed(self, board, player_num):
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    '''to check consecutive pieces, i.e. consecutive 2, 3, return how many consecutives exists'''
    def check(self, board, player_num, check_count):
        player_win_str_partial = '{0}'.format(player_num)
        player_win_str = str()
        for i in range(0, check_count):
            player_win_str = player_win_str + player_win_str_partial

        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            res = 0
            for row in b:
                if player_win_str in to_str(row):
                    res = res + 1
            return res

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            res = 0
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    res = res + 1

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            res = res + 1

            return res

        return (check_horizontal(board) +
                check_verticle(board) +
                check_diagonal(board))

    '''required API'''
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
        # Utility for non terminal states is computed using a linear evaluation function:
        # Eval(s) = 3(num) + 2(num) + num - (3(other_num) + 2(other_num) + other_num)

        other_number = 1
        if self.player_number == 1:
            other_number = 2

        # game ends
        win = self.game_completed(board, self.player_number)
        if win:
            return math.inf
        else:
            other_win = self.game_completed(board, other_number)
            if other_win:
                return -math.inf

        # game continue
        # ai_num
        # check 3
        ai_3 = self.check(board, self.player_number, 3)
        # check 2
        ai_2 = self.check(board, self.player_number, 2)
        # check 1
        ai_1 = self.check(board, self.player_number, 1)

        # other_num
        # check 3
        other_3 = self.check(board, other_number, 3)
        # check 2
        other_2 = self.check(board, other_number, 2)
        # check 1
        other_1 = self.check(board, other_number, 1)

        res = 3 * ai_3 + 2 * ai_2 + ai_1 - (3 * other_3 + 2 * other_2 + other_1)
        return res

    '''
    minmax is the basic algorithm of alpha_beta,
    since it only have two more parameters: alpha, beta
    param: maximizingPlayer: to max = true, to min = false
    '''
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        valid_locations = self.get_possible_moves(board).tolist()
        if depth == 0 or len(valid_locations) == 0 or self.game_completed(board, self.player_number):
            return None, self.evaluation_function(board)

        random.shuffle(valid_locations)

        if maximizing_player:
            best = -math.inf
            column = valid_locations[0]
            for column_tmp in valid_locations:
                board_tmp = copy.deepcopy(board)
                self.drop_piece(board_tmp, column_tmp, self.player_number)
                _, score = self.minimax(board_tmp, depth - 1, alpha, beta, False)
                best = max(best, score)
                if best > alpha:
                    alpha = best
                    column = column_tmp

                # Alpha Beta Pruning
                if alpha >= beta:
                    break

            return column, score

        else:  # Minimizing player
            column = valid_locations[0]
            best = math.inf
            for column_tmp in valid_locations:
                board_tmp = copy.deepcopy(board)
                self.drop_piece(board_tmp, column_tmp, self.opponent_number)
                _, score = self.minimax(board_tmp, depth - 1, alpha, beta, True)
                best = min(best, score)
                if best < beta:
                    beta = best
                    column = column_tmp

                # Alpha Beta Pruning
                if alpha >= beta:
                    break

            return column, score

    '''required API'''
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
        print("alpha_beta")
        col, _ = self.minimax(board, 3, -math.inf, math.inf, True)  # 3 is depth
        return col

    '''
    expectiminimax is the basic algorithm of expectimax,
    '''
    def expectiminimax(self, board, number, depth, node):
        # default: number is 2
        opp_number = 1
        if number == 1:
            opp_number = 2

        valid_locations = self.get_possible_moves(board)
        is_ended = self.game_completed(board, number)
        if depth == 0 or is_ended:
            return None, self.evaluation_function(board)

        if node: # our move
            alpha = -math.inf
            column = valid_locations[0]
            for col in valid_locations:
                row = self.get_drop_row(board, col)
                b_copy = copy.deepcopy(board)
                self.drop_piece(b_copy, row, col, number)
                new_score = self.expectiminimax(b_copy, number, depth - 1, False)[1]
                if new_score > alpha:
                    alpha = new_score
                    column = col

        else:  # random node
            alpha = 0
            column = valid_locations[0]
            for col in valid_locations:
                row = self.get_drop_row(board, col)
                b_copy = copy.deepcopy(board)
                self.drop_piece(b_copy, row, col, number)
                alpha = alpha + ((1.0/7.0) * self.expectiminimax(b_copy, opp_number, depth - 1, True)[1])
        return column, alpha

    '''required API'''
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
        print("expectimax")
        piece = self.player_number
        col, score = self.expectiminimax(board, piece, 2, True)
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
            if 0 in board[:,col]:
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




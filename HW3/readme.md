# three required API

## def evaluation_function(self, board)

## def get_alpha_beta_move(self, board)

## def get_expectimax_move(self, board)


get_alpha_beta_move(self, board) will call:<br>
def minimax(self, board, piece, depth, alpha, beta, maximizingPlayer)

def get_expectimax_move(self, board) will call:<br>
def expectiminimax(self, board, piece, depth, node)

evaluation_function:<br>
piece means the number of player, since we have two player, player1 and 2 <br>
1 find if current is terminate node<br>
2 find children node: find possible columns and find exact row in that columns<br>
3 use check function to find if exists consecutive 

confusing about evaluation since passing parameters are confusing now...


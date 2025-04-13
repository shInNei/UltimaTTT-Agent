import numpy as np
from state import UltimateTTT_Move, State
from state import State, State_2
from collections import defaultdict
import time
import copy

TIME_OUT = 5.5
start_time = 0
transposition_table = defaultdict(lambda: {"score": None, "depth": -1, "flag": None})

def check_win_condition(board):
    row_sum = np.sum(board, 1)
    col_sum = np.sum(board, 0)
    diag_sum_top_left = board.trace()
    diag_sum_top_right = board[::-1].trace()

    player_one_wins = any(row_sum == 3) + any(col_sum == 3)
    player_one_wins += (diag_sum_top_left == 3) + (diag_sum_top_right == 3)

    if player_one_wins:
        return 1

    player_two_wins = any(row_sum == -3) + any(col_sum == -3)
    player_two_wins += (diag_sum_top_left == -3) + (diag_sum_top_right == -3)

    if player_two_wins:
        return -1

    return 0


def real_evaluate_position(board, row, col, player):
    evaluation = 0
    points = [0.2, 0.17, 0.2, 0.17, 0.22, 0.17, 0.2, 0.17, 0.2]

    board[row, col] = player
    evaluation += player * points[3 * row + col]

    if board[0, 0] + board[0, 1] + board[0, 2] == 2 * player or board[1, 0] + board[1, 1] + board[1, 2] == 2 * player or \
            board[2, 0] + \
            board[2, 1] + board[2, 2] == 2 * player:
        evaluation += player * 1
    if board[0, 0] + board[1, 0] + board[2, 0] == 2 * player or board[0, 1] + board[1, 1] + board[2, 1] == 2 * player or \
            board[0, 2] + \
            board[1, 2] + board[2, 2] == 2 * player:
        evaluation += player * 1
    if board[0, 0] + board[1, 1] + board[2, 2] == 2 * player or board[0, 2] + board[1, 1] + board[2, 0] == 2 * player:
        evaluation += player * 1

    if board[0, 0] + board[0, 1] + board[0, 2] == 3 * player or board[1, 0] + board[1, 1] + board[1, 2] == 3 * player or \
            board[2, 0] + \
            board[2, 1] + board[2, 2] == 3 * player:
        evaluation += player * 5
    if board[0, 0] + board[1, 0] + board[2, 0] == 3 * player or board[0, 1] + board[1, 1] + board[2, 1] == 3 * player or \
            board[0, 2] + \
            board[1, 2] + board[2, 2] == 3 * player:
        evaluation += player * 5
    if board[0, 0] + board[1, 1] + board[2, 2] == 3 * player or board[0, 2] + board[1, 1] + board[2, 0] == 3 * player:
        evaluation += player * 5

    board[row, col] = -player

    if board[0, 0] + board[0, 1] + board[0, 2] == -3 * player or board[1, 0] + board[1, 1] + board[
        1, 2] == -3 * player or board[2, 0] + \
            board[2, 1] + board[2, 2] == -3 * player:
        evaluation += player * 2
    if board[0, 0] + board[1, 0] + board[2, 0] == -3 * player or board[0, 1] + board[1, 1] + board[
        2, 1] == -3 * player or board[0, 2] + \
            board[1, 2] + board[2, 2] == -3 * player:
        evaluation += player * 2
    if board[0, 0] + board[1, 1] + board[2, 2] == -3 * player or board[0, 2] + board[1, 1] + board[2, 0] == -3 * player:
        evaluation += player * 2

    board[row, col] = player
    evaluation += check_win_condition(board) * 15

    board[row, col] = 0
    return evaluation


def real_evaluate_board(board):
    evaluation = 0
    points = [0.2, 0.17, 0.2, 0.17, 0.22, 0.17, 0.2, 0.17, 0.2]

    for cell in range(9):
        evaluation += board[cell // 3, cell % 3] * points[cell]

    if board[0, 0] + board[0, 1] + board[0, 2] == 2 or board[1, 0] + board[1, 1] + board[1, 2] == 2 or board[2, 0] + \
            board[2, 1] + board[2, 2] == 2:
        evaluation += 6
    if board[0, 0] + board[1, 0] + board[2, 0] == 2 or board[0, 1] + board[1, 1] + board[2, 1] == 2 or board[0, 2] + \
            board[1, 2] + board[2, 2] == 2:
        evaluation += 6
    if board[0, 0] + board[1, 1] + board[2, 2] == 2 or board[0, 2] + board[1, 1] + board[2, 0] == 2:
        evaluation += 7

    if (board[0, 0] + board[0, 1] == -2 and board[0, 2] == 1) or (
            board[0, 1] + board[0, 2] == -2 and board[0, 0] == 1) or (
            board[0, 0] + board[0, 2] == -2 and board[0, 1] == 1):
        evaluation += 9

    if (board[1, 0] + board[1, 1] == -2 and board[1, 2] == 1) or (
            board[1, 1] + board[1, 2] == -2 and board[1, 0] == 1) or (
            board[1, 0] + board[1, 2] == -2 and board[1, 1] == 1):
        evaluation += 9

    if (board[2, 0] + board[2, 1] == -2 and board[2, 2] == 1) or (
            board[2, 1] + board[2, 2] == -2 and board[2, 0] == 1) or (
            board[2, 0] + board[2, 2] == -2 and board[2, 1] == 1):
        evaluation += 9

    if (board[0, 0] + board[1, 0] == -2 and board[2, 0] == 1) or (
            board[1, 0] + board[2, 0] == -2 and board[0, 0] == 1) or (
            board[0, 0] + board[2, 0] == -2 and board[1, 0] == 1):
        evaluation += 9

    if (board[0, 1] + board[1, 1] == -2 and board[2, 1] == 1) or (
            board[1, 1] + board[2, 1] == -2 and board[0, 1] == 1) or (
            board[0, 1] + board[2, 1] == -2 and board[1, 1] == 1):
        evaluation += 9

    if (board[0, 2] + board[1, 2] == -2 and board[2, 2] == 1) or (
            board[1, 2] + board[2, 2] == -2 and board[0, 2] == 1) or (
            board[0, 2] + board[2, 2] == -2 and board[1, 2] == 1):
        evaluation += 9

    if (board[0, 0] + board[1, 1] == -2 and board[2, 2] == 1) or (
            board[1, 1] + board[2, 2] == -2 and board[0, 0] == 1) or (
            board[0, 0] + board[2, 2] == -2 and board[1, 1] == 1):
        evaluation += 9

    if (board[0, 2] + board[1, 1] == -2 and board[2, 0] == 1) or (
            board[1, 1] + board[2, 0] == -2 and board[0, 2] == 1) or (
            board[0, 2] + board[2, 0] == -2 and board[1, 1] == 1):
        evaluation += 9

    if board[0, 0] + board[0, 1] + board[0, 2] == -2 or board[1, 0] + board[1, 1] + board[1, 2] == -2 or board[2, 0] + \
            board[2, 1] + board[2, 2] == -2:
        evaluation -= 6
    if board[0, 0] + board[1, 0] + board[2, 0] == -2 or board[0, 1] + board[1, 1] + board[2, 1] == -2 or board[0, 2] + \
            board[1, 2] + board[2, 2] == -2:
        evaluation -= 6
    if board[0, 0] + board[1, 1] + board[2, 2] == -2 or board[0, 2] + board[1, 1] + board[2, 0] == -2:
        evaluation -= 7

    if (board[0, 0] + board[0, 1] == 2 and board[0, 2] == -1) or (
            board[0, 1] + board[0, 2] == 2 and board[0, 0] == -1) or (
            board[0, 0] + board[0, 2] == 2 and board[0, 1] == -1):
        evaluation -= 9

    if (board[1, 0] + board[1, 1] == 2 and board[1, 2] == -1) or (
            board[1, 1] + board[1, 2] == 2 and board[1, 0] == -1) or (
            board[1, 0] + board[1, 2] == 2 and board[1, 1] == -1):
        evaluation -= 9

    if (board[2, 0] + board[2, 1] == 2 and board[2, 2] == -1) or (
            board[2, 1] + board[2, 2] == 2 and board[2, 0] == -1) or (
            board[2, 0] + board[2, 2] == 2 and board[2, 1] == -1):
        evaluation -= 9

    if (board[0, 0] + board[1, 0] == 2 and board[2, 0] == -1) or (
            board[1, 0] + board[2, 0] == 2 and board[0, 0] == -1) or (
            board[0, 0] + board[2, 0] == 2 and board[1, 0] == -1):
        evaluation -= 9

    if (board[0, 1] + board[1, 1] == 2 and board[2, 1] == -1) or (
            board[1, 1] + board[2, 1] == 2 and board[0, 1] == -1) or (
            board[0, 1] + board[2, 1] == 2 and board[1, 1] == -1):
        evaluation -= 9

    if (board[0, 2] + board[1, 2] == 2 and board[2, 2] == -1) or (
            board[1, 2] + board[2, 2] == 2 and board[0, 2] == -1) or (
            board[0, 2] + board[2, 2] == 2 and board[1, 2] == -1):
        evaluation -= 9

    if (board[0, 0] + board[1, 1] == 2 and board[2, 2] == -1) or (
            board[1, 1] + board[2, 2] == 2 and board[0, 0] == -1) or (
            board[0, 0] + board[2, 2] == 2 and board[1, 1] == -1):
        evaluation -= 9

    if (board[0, 2] + board[1, 1] == 2 and board[2, 0] == -1) or (
            board[1, 1] + board[2, 0] == 2 and board[0, 2] == -1) or (
            board[0, 2] + board[2, 0] == 2 and board[1, 1] == -1):
        evaluation -= 9

    evaluation += check_win_condition(board) * 12
    return evaluation


def evaluation_function(state: State):
    eval_mul = [1.4, 1, 1.4, 1, 1.75, 1, 1.4, 1, 1.4]

    evaluation = 0
    for i in range(9):
        evaluation += 1.5 * real_evaluate_board(state.blocks[i]) * eval_mul[i]
        evaluation += state.global_cells[i] * eval_mul[i]

    result = check_win_condition(state.global_cells.reshape(3, 3))
    evaluation += result * 5000
    evaluation += real_evaluate_board(state.global_cells.reshape(3, 3)) * 150
    return evaluation

def recurse(state: State, depth, alpha, beta):
    global transposition_table
    
    #check transposition table
    hash_key = hash(repr(state))
    
    if transposition_table[hash_key]["depth"] >= depth:
        if transposition_table[hash_key]["flag"] == "exact":
            return transposition_table[hash_key]["score"]
        elif transposition_table[hash_key]["flag"] == "lowerbound":
            alpha = max(alpha, transposition_table[hash_key]["score"])
        elif transposition_table[hash_key]["flag"] == "upperbound":
            beta = min(beta, transposition_table[hash_key]["score"])
        if alpha >= beta:
            return transposition_table[hash_key]["score"]
        
    if state.game_over or depth == 0 or time.time() - start_time >= TIME_OUT:
        return evaluation_function(state)

    valid_moves = state.get_valid_moves
    if state.player_to_move == state.X:
        max_utility = -float('inf')
        for move in valid_moves:
            child_state = copy.deepcopy(state)
            child_state.act_move(move)
            utility = recurse(child_state, depth - 1, alpha, beta)
            if utility > max_utility:
                max_utility = utility
            if max_utility > alpha:
                alpha = max_utility
            if alpha >= beta:
                break
        # Store the result in the transposition table
        transposition_table[hash_key]["score"] = alpha
        transposition_table[hash_key]["depth"] = depth
        transposition_table[hash_key]["flag"] = "exact"
        return alpha
    else:
        min_utility = float('inf')
        for move in valid_moves:
            child_state = copy.deepcopy(state)
            child_state.act_move(move)
            utility = recurse(child_state, depth - 1, alpha, beta)
            if utility < min_utility:
                min_utility = utility
            if min_utility < beta:
                beta = min_utility
            if beta <= alpha:
                break
        # Store the result in the transposition table
        transposition_table[hash_key]["score"] = beta
        transposition_table[hash_key]["depth"] = depth
        transposition_table[hash_key]["flag"] = "exact"
        return beta


numMoves = 0


def select_move(cur_state, remain_time):
    global start_time
    start_time = time.time()
    
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) == 0:
        return None

    for move in valid_moves:
        child_state = copy.deepcopy(cur_state)
        child_state.act_move(move)
        if child_state.game_over is True:
            return move

    global numMoves

    if numMoves == 0 and cur_state.player_to_move == cur_state.X:
        numMoves += 1
        return UltimateTTT_Move(4, 1, 1, cur_state.X)

    scores = np.zeros(len(valid_moves))
    for i in range(len(valid_moves)):
        scores[i] += real_evaluate_position(cur_state.blocks[valid_moves[i].index_local_board], valid_moves[i].x,
                                            valid_moves[i].y, valid_moves[i].value) * 45

    for i in range(len(valid_moves)):
        child_state = None
        child_state = copy.deepcopy(cur_state)

        child_state.act_move(valid_moves[i])

        utility = 0
        alpha = -float('inf')
        beta = float('inf')
        if cur_state.free_move is True:
            if numMoves < 17:
                utility = recurse(child_state, 3, alpha, beta)
            elif numMoves < 20:
                utility = recurse(child_state, 4, alpha, beta)
            elif numMoves < 25:
                utility = recurse(child_state, 5, alpha, beta)
            else:
                utility = recurse(child_state, 6, alpha, beta)
        else:
            if numMoves < 15:
                utility = recurse(child_state, 4, alpha, beta)
            elif numMoves < 20:
                utility = recurse(child_state, 5, alpha, beta)
            else:
                utility = recurse(child_state, 6, alpha, beta)

        scores[i] += utility

    best_move = None
    if valid_moves[0].value == 1:
        best_score = -float('inf')
        for i in range(len(valid_moves)):
            if scores[i] > best_score:
                best_score = scores[i]
                best_move = valid_moves[i]
    else:
        best_score = float('inf')
        for i in range(len(valid_moves)):
            if scores[i] < best_score:
                best_score = scores[i]
                best_move = valid_moves[i]

    numMoves += 1
    return best_move

# MCTS agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from constants import *
from copy import deepcopy
import numpy as np
import sys
import time

from datetime import timedelta
from datetime import datetime
from random import choice
from math import log, sqrt

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1)) #moves and opposite of directions for helper functions
OPPOSITES = {0: 2, 1: 3, 2: 0, 3: 1}


def check_win(chess_board, my_pos, adv_pos):
    """check if the position has reached the ending 
    returns a tuple of a bool representing if the game is at an end state and the score
    the score is 1 if the first player wins, 0 otherwise.
    taken from the world class
    """
    board_size = chess_board.shape[0]
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                MOVES[1:3]
            ):  # Only check down and right
                if chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(tuple(my_pos))
    p1_r = find(tuple(adv_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, 0
    if p0_score == p1_score:
        return True, 0
    return True, max(0, (p0_score-p1_score)/abs(p0_score-p1_score))

def get_moves(chess_board, my_pos, adv_pos, max_step):
    """gets max_step*2 random moves from
    returns a list of random moves
    """
    moves = set()
    for i in range(max_step*4):
        moves.add(get_random_move(chess_board, my_pos, adv_pos, max_step))

    return list(moves)

def apply_move(chess_board, r, c, dir):
    """
    returns the position that results from applying the mvoe to the board, as a board (np array).
    from world class
    """
    board = deepcopy(chess_board)
    # Set the barrier to True
    board[r, c, dir] = True
    # Set the opposite barrier to True
    move = MOVES[dir]
    board[r + move[0], c + move[1], OPPOSITES[dir]] = True
    return board

def get_random_move(chess_board, my_pos, adv_pos, max_step):
    """returns a random move from the given position, as a r,c,dir tuple
    taken from the world class
    """
    # Moves (Up, Right, Down, Left)
    ori_pos = deepcopy(my_pos)
    steps = np.random.randint(0, max_step + 1)

    # Random Walk
    for _ in range(steps):
        r, c = my_pos
        dir = np.random.randint(0, 4)
        m_r, m_c = MOVES[dir]
        my_pos = (r + m_r, c + m_c)

        # Special Case enclosed by Adversary
        k = 0
        while chess_board[r, c, dir] or my_pos == adv_pos:
            k += 1
            if k > 300:
                break
            dir = np.random.randint(0, 4)
            m_r, m_c = MOVES[dir]
            my_pos = (r + m_r, c + m_c)

        if k > 300:
            my_pos = ori_pos
            break

    # Put Barrier
    dir = np.random.randint(0, 4)
    r, c = my_pos
    while chess_board[r, c, dir]:
        dir = np.random.randint(0, 4)

    return my_pos, dir    

"""
Code adapted from http://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
MonteCarlo Algorithm for the game playing
"""
class MonteCarlo:
    def __init__(self,chess_board, my_pos, adv_pos, max_step, **kwargs):
        #self.states = []#store encountered states
        self.max_step = max_step
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos

        sec = kwargs.get("time", 1.95)
        self.calculation_time = timedelta(seconds=sec) #time llowed for calculation
        self.setup = True #setup allows for more computation time, true until updated after the first computation
        self.max_moves = kwargs.get("max_moves", 50) #max moves for simulations
        self.C = kwargs.get("C", 1.4) #constant for confidence interval 
        self.wins = {} #store number of wins for each state
        self.plays ={} #store number of plays for each state
        self.num = 0 #move count

    def get_move(self, board, my_pos, adv_pos):
        """returns a move from the algorithm, the one with the best stats
        """
        self.num += 1 #one more move from the adv
        player = True #player is true or false, True when get move is called
        self.adv_pos = adv_pos#update values
        self.my_pos = my_pos
        self.chess_board = board
        moves = get_moves(self.chess_board, self.my_pos, self.adv_pos, self.max_step)#get list of random moves
        start = datetime.utcnow()
        if self.setup:
            time = timedelta(seconds = 29.95)#allow longer computation at setup
            self.setup=False
        else :
            time = self.calculation_time

        while datetime.utcnow() - start < time:#simulate until the time limit
            self.run_simulation()

        moves_states = [(play, (play, self.num)) for play in moves]#pack moves and states together

        move,S = moves_states[0]
        percent = self.wins.get((player,S), 0)/self.plays.get((player, S), 1)

        for p, S in moves_states[1:]:#update best move and state
            if self.wins.get((player,S), 0)/self.plays.get((player, S), 1) > percent:
                move = p
                percent = self.wins.get((player,S), 0)/self.plays.get((player, S), 1)

        self.num += 1#one more move
        return move

    def run_simulation(self):
        """runs simulations. Expands one state and adds it to the plays and wins dictionaries. 
        Chooses moves based on current statistics or at random if not possible.
        updates the statistics if a winner is reached.
        """

        visited_states = set() #store visited stated in this simulation
        player = True#start with our player
        num = self.num #store locally for faster lookup
        board = deepcopy(self.chess_board)
        pos = self.my_pos
        adv = self.adv_pos
        max_step = self.max_step
        plays, wins = self.plays, self.wins
        winner = None#winner is none until there is one

        expand = True
        for t in range(self.max_moves):#simulate until max_moves
            moves = get_moves(board, pos, adv, max_step)
            moves_states = [(play, (play, num)) for play in moves]

            if all(plays.get((player, S)) for p, S in moves_states):#if all the moves have already considered, base the next play on the stats
                log_total = log(sum(plays[(player, S)] for p,S in moves_states))

                value = 0
                play, state = moves_states[0]

                for p,S in moves_states[1:]:#find maximum
                    v = (wins[(player, S)]/plays[(player, S)])+self.C*sqrt(log_total/plays[(player, S)])#UCT calculation
                    if v > value:
                        value = v
                        play = p
                        state = S

            else: 
                play, state = choice(moves_states)

            
            move,dir = play
            r,c = move
            board = apply_move(board, r,c,dir)#apply the move 
            
            if expand and (player, state) not in self.plays:#expand only one state
                expand = False
                self.plays[(player, state)] = 0#add to plays and wins
                self.wins[(player, state)] = 0
                
            visited_states.add((player, state))

            win, score = check_win(board, pos, adv)
            if win:
                if score == 1:
                    winner = player
                else:
                    winner = not player#break if win, store the winner
                break
            
            pos = adv#update for next loop, different player so we reverse
            adv = move
            player = not player
            num += 1

        for player, state in visited_states:#update plays and wins if necessary
            if (player, state) not in self.plays:
                continue
            self.plays[(player, state)] += 1
            if winner is not None:
                if player == winner:
                    self.wins[(player, state)]+= 1
            

@register_agent("mcts_agent")
class MCTSAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MCTSAgent, self).__init__()
        self.name = "MCTSAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True #enable autoplay
        self.monte_carlo = None #initialize 

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        if self.monte_carlo is None:
            self.monte_carlo = MonteCarlo(chess_board, my_pos, adv_pos, max_step)#create monteCarlo object if not present

        move = self.monte_carlo.get_move(chess_board, my_pos, adv_pos)
        return move #self.monte_carlo.get_move(chess_board, my_pos, adv_pos)#return move from monte carlo


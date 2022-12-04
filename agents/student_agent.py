# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True

    def available_pos(self, chess_board, my_pos, adv_pos, max_step):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        step = 0
        cur_pos = my_pos
        state_queue = [(cur_pos, step)]
        visited = {tuple(cur_pos)}
        while state_queue:
            cur_pos, step = state_queue.pop(0)
            r, c = cur_pos
            if step == max_step:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue
                r1, c1 = move
                next_pos = (r1 + r, c + c1)
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, step + 1))  # type: ignore

        visited = sorted(visited, key=lambda x: self.mht_dis(x, adv_pos))
        return visited

    def score(self, score_chess_board, score_pos, max_step):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        step = 0
        cur_pos = score_pos
        state_queue = [(cur_pos, step)]
        visited = {tuple(cur_pos)}
        while state_queue:
            cur_pos, step = state_queue.pop(0)
            r, c = cur_pos
            if step == max_step:
                break
            for dir, move in enumerate(moves):
                # print(r,c,dir,score_chess_board[r, c, dir])
                if score_chess_board[r, c, dir]:
                    continue
                r1, c1 = move
                next_pos = (r1 + r, c + c1)
                if tuple(next_pos) in visited:
                    continue
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, step + 1))  # type: ignore
        # print(visited)
        return len(visited)

    def mht_dis(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2)

    def bar_dis(self, pos, chess_board):
        score = 0
        r, c = pos
        for i in [0, 1, 2, 3]:
            if chess_board[r, c, i] == True:
                score += 1
        return score

    def relative_dir_ideal(self, mypos, advpos):
        r1, c1 = mypos
        r2, c2 = advpos
        if r1 > r2 and c1 > c2:
            return [0, 3]
        elif r1 > r2 and c1 == c2:
            return [0]
        elif r1 > r2 and c1 < c2:
            return [0, 1]
        elif r1 == r2 and c1 > c2:
            return [3]
        elif r1 == r2 and c1 < c2:
            return [1]
        elif r1 < r2 and c1 > c2:
            return [2, 3]
        elif r1 < r2 and c1 == c2:
            return [2]
        elif r1 < r2 and c1 < c2:
            return [1, 2]
        else:
            return [-1]

    def relative_dir_worse(self, mypos, advpos):
        r1, c1 = mypos
        r2, c2 = advpos
        if r1 > r2 and c1 > c2:
            return [1, 2]
        elif r1 > r2 and c1 == c2:
            return [1, 2, 3]
        elif r1 > r2 and c1 < c2:
            return [2, 3]
        elif r1 == r2 and c1 > c2:
            return [0, 1, 2]
        elif r1 == r2 and c1 < c2:
            return [0, 2, 3]
        elif r1 < r2 and c1 > c2:
            return [0, 1]
        elif r1 < r2 and c1 == c2:
            return [0, 1, 3]
        elif r1 < r2 and c1 < c2:
            return [0, 3]
        else:
            return [-1]

    def check_endgame(self, size, chess_board, p0_pos, p1_pos):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        father = dict()
        for r in range(size):
            for c in range(size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(size):
            for c in range(size):
                for dir, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(size):
            for c in range(size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        if p0_score >= p1_score:
            player_win = 0
        elif p0_score < p1_score:
            player_win = 1
        return True, player_win

    def check_endgame1(self, size, chess_board, p0_pos, p1_pos):

        p0_score = self.score(chess_board, p0_pos, size * size)
        p1_score = self.score(chess_board, p1_pos, size * size)
        # print(p0_score, p1_score,p0_pos,dir)
        if p0_score == p1_score:
            return False, p0_score, p1_score
        player_win = None
        if p0_score > p1_score:
            player_win = 0
        elif p0_score < p1_score:
            player_win = 1
        return True, player_win

    def check_oneshot(self, visited, chess_board, size, adv_pos, dir_sequence):
        for pos in visited:
            r, c = pos
            for dir in dir_sequence(pos, adv_pos):

                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    result = self.check_endgame(size, new_chessboard, pos, adv_pos)
                    if result[0]:
                        if result[1] == 0:
                            return pos, dir
                        if result[1] == 1:
                            continue
        return None

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
        # dummy return

        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        size = len(chess_board)
        step = 0
        cur_pos = ori_pos
        state_queue = [(cur_pos, step)]
        visited = {tuple(cur_pos)}

        while state_queue:
            cur_pos, step = state_queue.pop(0)
            r, c = cur_pos
            if step == max_step:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue
                r1, c1 = move
                next_pos = (r1 + r, c + c1)
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, step + 1))  # type: ignore
                

        visited = sorted(visited, key=lambda x: self.mht_dis(x, adv_pos) + self.bar_dis(x, chess_board))
        for pos in visited:
            r, c = pos
            for dir in self.relative_dir_ideal(pos, adv_pos):
                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    if dir == 0:
                        new_chessboard[r - 1, c, 2] = True
                    if dir == 1:
                        new_chessboard[r, c + 1, 3] = True
                    if dir == 2:
                        new_chessboard[r + 1, c, 0] = True
                    if dir == 3:
                        new_chessboard[r, c - 1, 1] = True
                    result = self.check_endgame(size, new_chessboard, pos, adv_pos)
                    if result[0]:
                        if result[1] == 0:
                            #print("s1.1")
                            return pos, dir
                        if result[1] == 1:
                            continue
        for pos in visited:
            r, c = pos
            for dir in self.relative_dir_worse(pos, adv_pos):

                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    if dir == 0:
                        new_chessboard[r - 1, c, 2] = True
                    if dir == 1:
                        new_chessboard[r, c + 1, 3] = True
                    if dir == 2:
                        new_chessboard[r + 1, c, 0] = True
                    if dir == 3:
                        new_chessboard[r, c - 1, 1] = True
                    result = self.check_endgame(size, new_chessboard, pos, adv_pos)
                    if result[0]:
                        if result[1] == 0:
                            #print("s1.2")
                            return pos, dir
                        if result[1] == 1:
                            continue

        for pos in visited:
            r, c = pos
            for dir in self.relative_dir_ideal(pos, adv_pos):
                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    if dir == 0:
                        new_chessboard[r - 1, c, 2] = True
                    if dir == 1:
                        new_chessboard[r, c + 1, 3] = True
                    if dir == 2:
                        new_chessboard[r + 1, c, 0] = True
                    if dir == 3:
                        new_chessboard[r, c - 1, 1] = True
                    result = self.check_endgame(size, new_chessboard, pos, adv_pos)
                    if not result[0]:
                        adv_possible_pos = self.available_pos(new_chessboard, adv_pos, pos, max_step)
                        loss_count = 0
                        for a_pos in adv_possible_pos:
                            a_r, a_c = a_pos
                            for a_dir in self.relative_dir_ideal(a_pos, pos):
                                new_new_chessboard = deepcopy(new_chessboard)
                                if not new_new_chessboard[a_r, a_c, a_dir]:
                                    new_new_chessboard[a_r, a_c, a_dir] = True
                                    if a_dir == 0:
                                        new_new_chessboard[a_r - 1, a_c, 2] = True
                                    if a_dir == 1:
                                        new_new_chessboard[a_r, a_c + 1, 3] = True
                                    if a_dir == 2:
                                        new_new_chessboard[a_r + 1, a_c, 0] = True
                                    if a_dir == 3:
                                        new_new_chessboard[a_r, a_c - 1, 1] = True
                                    a_result = self.check_endgame(size, new_new_chessboard, pos, a_pos)
                                    if a_result[0]:
                                        if a_result[1] == 1:
                                            loss_count += 1
                        for a_pos in adv_possible_pos:
                            a_r, a_c = a_pos
                            for a_dir in self.relative_dir_worse(a_pos, pos):
                                new_new_chessboard = deepcopy(new_chessboard)
                                if not new_new_chessboard[a_r, a_c, a_dir]:
                                    new_new_chessboard[a_r, a_c, a_dir] = True
                                    if a_dir == 0:
                                        new_new_chessboard[a_r - 1, a_c, 2] = True
                                    if a_dir == 1:
                                        new_new_chessboard[a_r, a_c + 1, 3] = True
                                    if a_dir == 2:
                                        new_new_chessboard[a_r + 1, a_c, 0] = True
                                    if a_dir == 3:
                                        new_new_chessboard[a_r, a_c - 1, 1] = True
                                    a_result = self.check_endgame(size, new_new_chessboard, pos, a_pos)
                                    if a_result[0]:
                                        if a_result[1] == 1:
                                            loss_count += 1
                        if loss_count == 0:
                            return pos, dir

            for pos in visited:
                r, c = pos
                for dir in self.relative_dir_worse(pos, adv_pos):
                    if not chess_board[r, c, dir]:
                        new_chessboard = deepcopy(chess_board)
                        new_chessboard[r, c, dir] = True
                        if dir == 0:
                            new_chessboard[r - 1, c, 2] = True
                        if dir == 1:
                            new_chessboard[r, c + 1, 3] = True
                        if dir == 2:
                            new_chessboard[r + 1, c, 0] = True
                        if dir == 3:
                            new_chessboard[r, c - 1, 1] = True
                        result = self.check_endgame(size, new_chessboard, pos, adv_pos)
                        if not result[0]:
                            adv_possible_pos = self.available_pos(new_chessboard, adv_pos, pos, max_step)
                            loss_count = 0
                            for a_pos in adv_possible_pos:
                                a_r, a_c = a_pos
                                for a_dir in self.relative_dir_ideal(a_pos, pos):
                                    new_new_chessboard = deepcopy(new_chessboard)
                                    if not new_new_chessboard[a_r, a_c, a_dir]:
                                        new_new_chessboard[a_r, a_c, a_dir] = True
                                        if a_dir == 0:
                                            new_new_chessboard[a_r - 1, a_c, 2] = True
                                        if a_dir == 1:
                                            new_new_chessboard[a_r, a_c + 1, 3] = True
                                        if a_dir == 2:
                                            new_new_chessboard[a_r + 1, a_c, 0] = True
                                        if a_dir == 3:
                                            new_new_chessboard[a_r, a_c - 1, 1] = True
                                        a_result = self.check_endgame(size, new_new_chessboard, pos, a_pos)
                                        if a_result[0]:
                                            if a_result[1] == 1:
                                                loss_count += 1
                            for a_pos in adv_possible_pos:
                                a_r, a_c = a_pos
                                for a_dir in self.relative_dir_worse(a_pos, pos):
                                    new_new_chessboard = deepcopy(new_chessboard)
                                    if not new_new_chessboard[a_r, a_c, a_dir]:
                                        new_new_chessboard[a_r, a_c, a_dir] = True
                                        if a_dir == 0:
                                            new_new_chessboard[a_r - 1, a_c, 2] = True
                                        if a_dir == 1:
                                            new_new_chessboard[a_r, a_c + 1, 3] = True
                                        if a_dir == 2:
                                            new_new_chessboard[a_r + 1, a_c, 0] = True
                                        if a_dir == 3:
                                            new_new_chessboard[a_r, a_c - 1, 1] = True
                                        a_result = self.check_endgame(size, new_new_chessboard, pos, a_pos, )
                                        if a_result[0]:
                                            if a_result[1] == 1:
                                                loss_count += 1
                            if loss_count == 0:
                                return pos, dir
        for pos in visited:
            r, c = pos
            for dir in self.relative_dir_ideal(pos, adv_pos):
                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    if dir == 0:
                        new_chessboard[r - 1, c, 2] = True
                    if dir == 1:
                        new_chessboard[r, c + 1, 3] = True
                    if dir == 2:
                        new_chessboard[r + 1, c, 0] = True
                    if dir == 3:
                        new_chessboard[r, c - 1, 1] = True
                    result = self.check_endgame(size, new_chessboard, pos, adv_pos)
                    if not result[0]:
                        return pos, dir
        for pos in visited:
            r, c = pos
            for dir in self.relative_dir_worse(pos, adv_pos):
                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    if dir == 0:
                        new_chessboard[r - 1, c, 2] = True
                    if dir == 1:
                        new_chessboard[r, c + 1, 3] = True
                    if dir == 2:
                        new_chessboard[r + 1, c, 0] = True
                    if dir == 3:
                        new_chessboard[r, c - 1, 1] = True
                    result = self.check_endgame(size, new_chessboard, pos, adv_pos)
                    if not result[0]:
                        return pos, dir
        for pos in visited:
            r, c = pos
            for dir in self.relative_dir_ideal(pos, adv_pos):
                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    if dir == 0:
                        new_chessboard[r - 1, c, 2] = True
                    if dir == 1:
                        new_chessboard[r, c + 1, 3] = True
                    if dir == 2:
                        new_chessboard[r + 1, c, 0] = True
                    if dir == 3:
                        new_chessboard[r, c - 1, 1] = True
                    return pos, dir
        for pos in visited:
            r, c = pos
            for dir in self.relative_dir_worse(pos, adv_pos):
                if not chess_board[r, c, dir]:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[r, c, dir] = True
                    if dir == 0:
                        new_chessboard[r - 1, c, 2] = True
                    if dir == 1:
                        new_chessboard[r, c + 1, 3] = True
                    if dir == 2:
                        new_chessboard[r + 1, c, 0] = True
                    if dir == 3:
                        new_chessboard[r, c - 1, 1] = True
                    return pos, dir
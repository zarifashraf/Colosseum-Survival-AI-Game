from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy


@register_agent("student_agent")
class StudentAgent(Agent):

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

    # helper function 1
    def possibleSteps(self, chess_board, my_pos, adv_pos, max_step, calc_barrier):
        moves, stepCount, current_pos = ((-1, 0), (0, 1), (1, 0), (0, -1)), 0, my_pos  # U, R, D, L, step, position
        pos_list, tested = [(current_pos, stepCount)], {tuple(current_pos)}

        while len(pos_list) != 0:
            current_pos, stepCount = pos_list.pop(0)
            # row, col = current_pos
            row = current_pos[0]
            col = current_pos[1]

            if stepCount >= max_step:
                break

            def check_next_move():
                for direction, move in enumerate(moves):
                    if chess_board[row, col, direction] == True:
                        continue

                    row_step = move[0]
                    col_step = move[1]
                    next_row, next_col = row_step + row, col_step + col
                    next_pos = (next_row, next_col)

                    if tuple(next_pos) in tested:
                        continue

                    if next_pos == adv_pos:
                        continue

                    tested.add(tuple(next_pos))
                    pos_list.append((next_pos, stepCount + 1))  # type: ignore

            check_next_move()

        dist_tested = []
        sorted_tested = list(tested.copy())
        index = 0
        index_2 = 0

        for positions in tested:
            if calc_barrier:
                dist_position = self.calculateDistance(positions, adv_pos) + self.calculateBarrierNumbers(positions, chess_board)
            else:
                dist_position = self.calculateDistance(positions, adv_pos)
            dist_tested.append((index, dist_position))
            index += 1

        sorted_dist_tested = sorted(dist_tested, key=lambda distance: distance[1])

        for (ind, dist) in sorted_dist_tested:
            sorted_tested[index_2] = list(tested)[ind]
            index_2 += 1

        return sorted_tested

    # Calculates the distance between two positions
    def calculateDistance(self, pos_1, pos_2):
        return abs(pos_2[0] - pos_1[0]) + abs(pos_2[1] - pos_1[1])

    # Caculate the number of barriers around a given position
    def calculateBarrierNumbers(self, pos, chess_board):
        barriers = 0
        for i in range(4):
            if chess_board[pos[0], pos[1], i]:
                barriers += 1
        return barriers

    # returns 1 if var 1 > var 2 & -1 if var 1 < var 2, otherwise 0
    def relativePositioning(self, var1, var2):

        if (var1 - var2) > 0:
            return 1
        elif (var1 - var2) < 0:
            return -1
        else:
            return 0

    # Calculating the relative position of the adversary 
    def possible_adv_dir(self, my_pos, adv_pos, is_ideal):
        result = []

        adv_pos_dict = dict()
        if is_ideal:
            adv_pos_dict = dict({(1, 1): [0, 3],
                                 (1, 0): [0],
                                 (1, -1): [0, 1],
                                 (0, 1): [3],
                                 (0, -1): [1],
                                 (-1, 1): [2, 3],
                                 (-1, 0): [2],
                                 (-1, -1): [1, 2]})
        else:
            adv_pos_dict = dict({(1, 1): [1, 2],
                                 (1, 0): [1, 2, 3],
                                 (1, -1): [2, 3],
                                 (0, 1): [0, 1, 2],
                                 (0, -1): [0, 2, 3],
                                 (-1, 1): [0, 1],
                                 (-1, 0): [0, 1, 3],
                                 (-1, -1): [0, 3]})

        my_row, my_col = my_pos
        adv_row, adv_col = adv_pos

        key_checker = tuple((self.relativePositioning(my_row, adv_row), self.relativePositioning(my_col, adv_col)));

        for key in adv_pos_dict.keys():
            if key_checker == key:
                result = adv_pos_dict.get(key)
                break

        return result

    # FUNCTION TAKE FROM WORLD.PY
    def check_endgame(self, chess_board, p0_pos, p1_pos):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
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
                        moves[1:3]
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

    def check_oneshot(self, visited, chess_board, adv_pos):
        for position in visited:
            row = position[0]
            col = position[1]

            for direction in self.possible_adv_dir(position, adv_pos, True):
                if chess_board[row, col, direction] == False:
                    new_chessboard = deepcopy(chess_board)
                    new_chessboard[row, col, direction] = True
                    if direction == 0:
                        new_chessboard[row - 1, col, 2] = True
                    if direction == 1:
                        new_chessboard[row, col + 1, 3] = True
                    if direction == 2:
                        new_chessboard[row + 1, col, 0] = True
                    if direction == 3:
                        new_chessboard[row, col - 1, 1] = True
                    result = self.check_endgame(new_chessboard, position, adv_pos)
                    if result[0]:
                        if result[1] == 0:
                            # print("s1.1")
                            return position, direction
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
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        size = len(chess_board)
        step = 0
        cur_pos = ori_pos
        state_queue = [(cur_pos, step)]
        visited = {tuple(cur_pos)}

        visited = self.possibleSteps(chess_board, my_pos, adv_pos, max_step, True)

        result = self.check_oneshot(visited, chess_board, adv_pos)

        if result is not None:
            pos, dir = result
            return pos, dir

        for pos in visited:
            r, c = pos
            for dir in self.possible_adv_dir(pos, adv_pos, False):

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
                    result = self.check_endgame(new_chessboard, pos, adv_pos)
                    if result[0]:
                        if result[1] == 0:
                            #print("s1.2")
                            return pos, dir
                        if result[1] == 1:
                            continue

        for pos in visited:
            r, c = pos
            for dir in self.possible_adv_dir(pos, adv_pos, True):
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
                    result = self.check_endgame(new_chessboard, pos, adv_pos)
                    if not result[0]:
                        adv_possible_pos = self.possibleSteps(new_chessboard, adv_pos, pos, max_step, False)
                        loss_count = 0
                        for a_pos in adv_possible_pos:
                            a_r, a_c = a_pos
                            for a_dir in self.possible_adv_dir(a_pos, pos, True):
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
                                    a_result = self.check_endgame(new_new_chessboard, pos, a_pos)
                                    if a_result[0]:
                                        if a_result[1] == 1:
                                            loss_count += 1
                        for a_pos in adv_possible_pos:
                            a_r, a_c = a_pos
                            for a_dir in self.possible_adv_dir(a_pos, pos, False):
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
                                    a_result = self.check_endgame(new_new_chessboard, pos, a_pos)
                                    if a_result[0]:
                                        if a_result[1] == 1:
                                            loss_count += 1
                        if loss_count == 0:
                            return pos, dir

            for pos in visited:
                r, c = pos
                for dir in self.possible_adv_dir(pos, adv_pos, False):
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
                        result = self.check_endgame(new_chessboard, pos, adv_pos)
                        if not result[0]:
                            adv_possible_pos = self.possibleSteps(new_chessboard, adv_pos, pos, max_step, False)
                            loss_count = 0
                            for a_pos in adv_possible_pos:
                                a_r, a_c = a_pos
                                for a_dir in self.possible_adv_dir(a_pos, pos, True):
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
                                        a_result = self.check_endgame(new_new_chessboard, pos, a_pos)
                                        if a_result[0]:
                                            if a_result[1] == 1:
                                                loss_count += 1
                            for a_pos in adv_possible_pos:
                                a_r, a_c = a_pos
                                for a_dir in self.possible_adv_dir(a_pos, pos, False):
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
                                        a_result = self.check_endgame(new_new_chessboard, pos, a_pos, )
                                        if a_result[0]:
                                            if a_result[1] == 1:
                                                loss_count += 1
                            if loss_count == 0:
                                return pos, dir
        for pos in visited:
            r, c = pos
            for dir in self.possible_adv_dir(pos, adv_pos, True):
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
                    result = self.check_endgame(new_chessboard, pos, adv_pos)
                    if not result[0]:
                        return pos, dir
        for pos in visited:
            r, c = pos
            for dir in self.possible_adv_dir(pos, adv_pos, False):
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
                    result = self.check_endgame(new_chessboard, pos, adv_pos)
                    if not result[0]:
                        return pos, dir
        for pos in visited:
            r, c = pos
            for dir in self.possible_adv_dir(pos, adv_pos, True):
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
            for dir in self.possible_adv_dir(pos, adv_pos, False):
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
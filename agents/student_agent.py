from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy

# Inspiration from https://www.rebellionresearch.com/what-is-monte-carlo-tree-search-used-for
# @author Abrar Fahad Rahman Anik, Zarif Ashraf Zidane

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

    def finishing_move(self, attempted, chess_board, adv_pos, is_ideal):

        for position in attempted:
            row = position[0]
            col = position[1]

            dir_dict = dict({0: [-1, 0, 2],
                             1: [0, 1, 2],
                             2: [1, 0, -2],
                             3: [0, -1, -2]})

            dir_list = self.possible_adv_dir(position, adv_pos, is_ideal)

            for direction in dir_list:
                list1 = [row, col, direction]
                if chess_board[row, col, direction] == False:
                    chessboard_copy = deepcopy(chess_board)
                    chessboard_copy[row, col, direction] = True

                    for key in dir_dict.keys():
                        if direction == key:
                            list2 = dir_dict.get(key)
                            res_list = [list1[i] + list2[i] for i in range(len(list2))]
                            break

                    res_row, res_col, res_dir = res_list
                    chessboard_copy[res_row, res_col, res_dir] = True
                    result = self.check_endgame(chessboard_copy, position, adv_pos)

                    if result[0] == True:
                        if result[1] == 0:
                            return position, direction
                        else:
                            continue

        return None

    def path_planner(self, attempted, chess_board, adv_pos, max_step, is_ideal):

        for position in attempted:
            row = position[0]
            col = position[1]

            dir_dict = dict({0: [-1, 0, 2],
                             1: [0, 1, 2],
                             2: [1, 0, -2],
                             3: [0, -1, -2]})

            dir_list = self.possible_adv_dir(position, adv_pos, is_ideal)

            for direction in dir_list:
                list1 = [row, col, direction]
                if chess_board[row, col, direction] == False:
                    chessboard_copy = deepcopy(chess_board)
                    chessboard_copy[row, col, direction] = True

                    for key in dir_dict.keys():
                        if direction == key:
                            list2 = dir_dict.get(key)
                            res_list = [list1[i] + list2[i] for i in range(len(list2))]
                            break

                    res_row, res_col, res_dir = res_list
                    chessboard_copy[res_row, res_col, res_dir] = True

                    result = self.check_endgame(chessboard_copy, position, adv_pos)
                    if not result[0]:
                        adv_possible_pos = self.possibleSteps(chessboard_copy, adv_pos, position, max_step, False)
                        loss_count = 0

                        for opp_pos in adv_possible_pos:
                            opp_row = position[0]
                            opp_col = position[1]

                            opp_dir_dict = dict({0: [-1, 0, 2],
                                             1: [0, 1, 2],
                                             2: [1, 0, -2],
                                             3: [0, -1, -2]})

                            opp_dir_list = self.possible_adv_dir(opp_pos, position, True)

                            for opp_direction in opp_dir_list:
                                opp_list1 = [row, col, direction]
                                opp_chessboard = deepcopy(chessboard_copy)
                                if opp_chessboard[opp_row, opp_col, opp_direction] == False:
                                    opp_chessboard[opp_row, opp_col, opp_direction] = True

                                    for key in opp_dir_dict.keys():
                                        if direction == key:
                                            opp_list2 = opp_dir_dict.get(key)
                                            opp_res_list = [opp_list1[i] + opp_list2[i] for i in range(len(opp_list2))]
                                            break

                                    opp_res_row, opp_res_col, opp_res_dir = opp_res_list
                                    opp_chessboard[opp_res_row, opp_res_col, opp_res_dir] = True

                                    opp_result = self.check_endgame(opp_chessboard, position, opp_pos)

                                    if opp_result[0]:
                                        if opp_result[1] == 1:
                                            loss_count += 1

                        for opp_pos in adv_possible_pos:
                                opp_row = position[0]
                                opp_col = position[1]

                                opp_dir_dict = dict({0: [-1, 0, 2],
                                                   1: [0, 1, 2],
                                                   2: [1, 0, -2],
                                                   3: [0, -1, -2]})

                                opp_dir_list = self.possible_adv_dir(opp_pos, position, False)

                                for opp_direction in opp_dir_list:
                                    opp_list1 = [row, col, direction]
                                    opp_chessboard = deepcopy(chessboard_copy)
                                    if opp_chessboard[opp_row, opp_col, opp_direction] == False:
                                        opp_chessboard[opp_row, opp_col, opp_direction] = True

                                        for key in opp_dir_dict.keys():
                                            if direction == key:
                                                opp_list2 = opp_dir_dict.get(key)
                                                opp_res_list = [opp_list1[i] + opp_list2[i] for i in range(len(opp_list2))]
                                                break

                                        opp_res_row, opp_res_col, opp_res_dir = opp_res_list
                                        opp_chessboard[opp_res_row, opp_res_col, opp_res_dir] = True

                                    opp_result = self.check_endgame(opp_chessboard, position, opp_pos)

                                    if opp_result[0]:
                                        if opp_result[1] == 1:
                                            loss_count += 1
                        if loss_count == 0:
                            return position, direction

    def last_resort(self, attempted, chess_board, adv_pos, is_ideal, last_effort):

        for position in attempted:
            row = position[0]
            col = position[1]

            dir_dict = dict({0: [-1, 0, 2],
                             1: [0, 1, 2],
                             2: [1, 0, -2],
                             3: [0, -1, -2]})
            dir_list = self.possible_adv_dir(position, adv_pos, is_ideal)

            for direction in dir_list:
                list1 = [row, col, direction]
                if chess_board[row, col, direction] == False:
                    chessboard_copy = deepcopy(chess_board)
                    chessboard_copy[row, col, direction] = True

                    for key in dir_dict.keys():
                        if direction == key:
                            list2 = dir_dict.get(key)
                            res_list = [list1[i] + list2[i] for i in range(len(list2))]
                            break

                    res_row, res_col, res_dir = res_list
                    chessboard_copy[res_row, res_col, res_dir] = True

                    if not last_effort:
                        result = self.check_endgame(chessboard_copy, position, adv_pos)
                        if not result[0]:
                            return position, dir
                    else:
                        return position, dir

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

        attempted = self.possibleSteps(chess_board, my_pos, adv_pos, max_step, True)

        # CHECK FOR FINISHING MOVE ====================================================================================

        result = self.finishing_move(attempted, chess_board, adv_pos, True)

        if result is not None:
            position, direction = result
            return position, direction

        result = self.finishing_move(attempted, chess_board, adv_pos, False)

        if result is not None:
            position, direction = result
            return position, direction

        # CHECK FOR FINISHING MOVE ENDS ===============================================================================

        # CALCULATE MOVE N STEPS  =====================================================================================

        path_plan = self.path_planner(attempted, chess_board, adv_pos, max_step, True)

        if path_plan is not None:
            position, direction = path_plan
            return position, direction

        path_plan = self.path_planner(attempted, chess_board, adv_pos, max_step, False)

        if path_plan is not None:
            position, direction = path_plan
            return position, direction

        # CALCULATE MOVE N STEPS ENDS =================================================================================

        # LOSING SCENARIO ==============================================================================================

        next_move = self.last_resort(attempted, chess_board, adv_pos, True, False)

        if next_move is not None:
            position, direction = next_move
            return position, direction

        next_move = self.last_resort(attempted, chess_board, adv_pos, False, False)

        if next_move is not None:
            position, direction = next_move
            return position, direction

        next_move = self.last_resort(attempted, chess_board, adv_pos, True, True)

        if next_move is not None:
            position, direction = next_move
            return position, direction

        next_move = self.last_resort(attempted, chess_board, adv_pos, False, True)

        if next_move is not None:
            position, direction = next_move
            return position, direction

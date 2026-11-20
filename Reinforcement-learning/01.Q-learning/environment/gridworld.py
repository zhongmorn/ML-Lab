from run.configs import CONFIG

MAZE = CONFIG["maze"]
LAYOUT = tuple(MAZE["layout"])
SIZE = len(LAYOUT)
if not SIZE or any(len(row) != SIZE for row in LAYOUT):
    raise ValueError("迷宫必须为非空方形。")

cells = "".join(LAYOUT)
if cells.count("S") != 1 or cells.count("G") != 1:
    raise ValueError("迷宫必须各有一个起点 S 和终点 G。")

# 把按行展开的位置换算成 (行, 列)。
START = divmod(cells.index("S"), SIZE)
GOAL = divmod(cells.index("G"), SIZE)
WALLS = {divmod(index, SIZE) for index, cell in enumerate(cells) if cell == "#"}
HAZARDS = {divmod(index, SIZE) for index, cell in enumerate(cells) if cell == "X"}

SLIP_PROBABILITY = MAZE["slip_probability"]
STEP_REWARD = MAZE["step_reward"]
# 按上、右、下、左的顺序定义动作。
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


def move(state, action):
    row, column = state
    dr, dc = MOVES[action]
    next_row, next_column = row + dr, column + dc
    next_state = (next_row, next_column)
    if 0 <= next_row < SIZE and 0 <= next_column < SIZE and next_state not in WALLS:
        return next_state
    return state


def step(state, action, rng, *, step_reward=STEP_REWARD):
    # 发生打滑时，等概率改走原方向左侧或右侧的垂直方向。
    draw = rng.random()
    if draw < SLIP_PROBABILITY:
        turn = -1 if draw < SLIP_PROBABILITY / 2 else 1
        action = (action + turn) % len(MOVES)

    next_state = move(state, action)
    # 吃到食物或踩到陷阱时，只发放对应的终止奖励。
    if next_state == GOAL:
        return next_state, MAZE["food_reward"], True
    if next_state in HAZARDS:
        return next_state, MAZE["hazard_reward"], True
    return next_state, step_reward, False

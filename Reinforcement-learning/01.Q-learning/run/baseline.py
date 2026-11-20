from collections import deque

from environment.gridworld import GOAL, HAZARDS, MOVES, move


def shortest_path_policy():
    # 从食物反向做广度优先搜索。
    distances = {GOAL: 0}
    frontier = deque([GOAL])
    actions = range(len(MOVES))

    while frontier:
        state = frontier.popleft()
        for action in actions:
            neighbor = move(state, action)
            if neighbor not in HAZARDS and neighbor not in distances:
                distances[neighbor] = distances[state] + 1
                frontier.append(neighbor)

    # 每一步都从实际所在的格子选最短方向，打滑后也能重新选路。
    def select_action(state):
        return min(actions, key=lambda action: distances.get(move(state, action), float("inf")))

    return select_action

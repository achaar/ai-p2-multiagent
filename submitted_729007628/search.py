# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def DFSUtil(problem, v, visited, info, steps):
    # Mark the current node as visited
    # and print it
    from game import Directions
    e = Directions.EAST
    w = Directions.WEST
    n = Directions.NORTH
    s = Directions.SOUTH

    visited[v] = info
    steps.push(v)
    # print('entered ', v)
    if problem.isGoalState(v):  # goal state reached
        actions = []
        crawl = v
        # while steps.isEmpty() is False:
        while crawl != (visited[crawl])[0]:  # if the node is not equal to it's parent
            # node = steps.pop()
            node = crawl
            # print('result',node)
            parent = (visited[node])[0]
            direction = (visited[node])[1][1]
            if direction is 'South':
                direction = s
            elif direction is 'North':
                direction = n
            elif direction is 'East':
                direction = e
            elif direction is 'West':
                direction = w

            crawl = parent
            actions = actions + [direction]

        # print(actions)
        # print(list(reversed(actions)))
        return list(reversed(actions))

    # traverse all the nodes that are successors
    for i in reversed(
            problem.getSuccessors(v)):  # reversed order gives better path, because redundant paths are avoided
        node = i[0]
        if node not in visited:
            actions = DFSUtil(problem, i[0], visited, [v, i], steps)
            if actions:
                return actions
    # print('deleted ', v)
    visited.pop(v)
    var = steps.pop()
    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    Start: (5, 5)
    Is the start a goal? False
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    """
    "*** YOUR CODE HERE ***"
    '''
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

'''
    from util import Stack
    steps = Stack()
    visited = {}
    start_state = problem.getStartState()
    info = [start_state, (start_state, 'South', 1)]
    actions = DFSUtil(problem, problem.getStartState(), visited, info, steps)
    return actions


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from util import Queue
    from game import Directions
    e = Directions.EAST
    w = Directions.WEST
    n = Directions.NORTH
    s = Directions.SOUTH
    q_arr = Queue()
    visited = {}
    start_state = problem.getStartState()
    info = [start_state, (start_state, 'South', 1)]
    q_arr.push(info)
    while q_arr.isEmpty() is False:
        node_info = q_arr.pop()
        # print(node_info)
        node = node_info[1][0]  # info about the present node
        # print('processing node ', node)
        if problem.isGoalState(node):
            visited[node] = node_info
            actions = []
            crawl = node
            # print('Goal Reached: ', crawl)
            while crawl != (visited[crawl])[0]:  # if the node is not equal to it's parent
                parent = (visited[crawl])[0]
                direction = (visited[crawl])[1][1]
                if direction is 'South':
                    direction = s
                elif direction is 'North':
                    direction = n
                elif direction is 'East':
                    direction = e
                elif direction is 'West':
                    direction = w

                crawl = parent
                actions = actions + [direction]

            # print(list(reversed(actions)))
            return list(reversed(actions))
            # return the list of actions

        if node not in visited:  # process the children
            visited[node] = node_info
            # print('Successors: ', problem.getSuccessors(node))
            for i in problem.getSuccessors(node):
                child_info = [node, i]
                q_arr.push(child_info)

    return []

    # actions = DFSUtil(problem, problem.getStartState(), visited, info, steps)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    from util import PriorityQueue
    from game import Directions
    e = Directions.EAST
    w = Directions.WEST
    n = Directions.NORTH
    s = Directions.SOUTH
    # print('class',problem.__class__.__name__)
    q_arr = PriorityQueue()
    visited = {}
    start_state = problem.getStartState()
    info = [start_state, (start_state, 'South', 1), int(1)]  # parent, (self info), cost till now
    q_arr.push(info, info[2])
    while q_arr.isEmpty() is False:
        node_info = q_arr.pop()
        # print(node_info)
        node = node_info[1][0]  # info about the present node
        # print('processing node ', node)
        if problem.isGoalState(node):
            visited[node] = node_info
            actions = []
            crawl = node
            # print('Goal Reached: ', crawl)
            while crawl != (visited[crawl])[0]:  # if the node is not equal to it's parent
                parent = (visited[crawl])[0]
                direction = (visited[crawl])[1][1]
                if direction is 'South':
                    direction = s
                elif direction is 'North':
                    direction = n
                elif direction is 'East':
                    direction = e
                elif direction is 'West':
                    direction = w

                crawl = parent
                actions = actions + [direction]

            # print(list(reversed(actions)))
            return list(reversed(actions))
            # return the list of actions

        if node not in visited:  # process the children
            visited[node] = node_info
            # print('Successors: ', problem.getSuccessors(node))
            for i in problem.getSuccessors(node):
                cost = float(i[2]) + node_info[2]
                child_info = [node, i, cost]
                # child_info[1][2] = child_info[1][2] + node_info[1][2]
                q_arr.push(child_info, child_info[2])  # Add costs till that node
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    from util import PriorityQueue
    from game import Directions
    e = Directions.EAST
    w = Directions.WEST
    n = Directions.NORTH
    s = Directions.SOUTH
    # print('class', problem.__class__.__name__)
    q_arr = PriorityQueue()
    visited = {}
    start_state = problem.getStartState()
    info = [start_state, (start_state, 'South', 1), int(1)]  # parent, (self info), cost till now
    q_arr.push(info, info[2])
    while q_arr.isEmpty() is False:
        node_info = q_arr.pop()
        # print(node_info)
        node = node_info[1][0]  # info about the present node
        # print('processing node ', node)
        if problem.isGoalState(node):
            visited[node] = node_info
            actions = []
            crawl = node
            # print('Goal Reached: ', crawl)
            while crawl != (visited[crawl])[0]:  # if the node is not equal to it's parent
                parent = (visited[crawl])[0]
                direction = (visited[crawl])[1][1]
                if direction is 'South':
                    direction = s
                elif direction is 'North':
                    direction = n
                elif direction is 'East':
                    direction = e
                elif direction is 'West':
                    direction = w

                crawl = parent
                actions = actions + [direction]

            # print(list(reversed(actions)))
            return list(reversed(actions))
            # return the list of actions

        if node not in visited:  # process the children
            visited[node] = node_info
            # print('Successors: ', problem.getSuccessors(node))
            for i in problem.getSuccessors(node):
                cost = float(i[2]) + node_info[2]
                h_value = heuristic(i[0], problem)
                # print('f(n): ',cost)
                child_info = [node, i, cost]
                # child_info[1][2] = child_info[1][2] + node_info[1][2]
                q_arr.push(child_info, child_info[2] + h_value)  # Add costs till that node

                # if h_value>cost:
                #     print('cost: ', cost, 'heuristic: ', h_value)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

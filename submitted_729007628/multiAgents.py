# multiAgents.py
# --------------
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


from util import manhattanDistance, PriorityQueue
from game import Directions
import random, util

from game import Agent
from searchAgents import mazeDistance

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # print('+++++++++++++++++++++++++++++++++')
        # print('current pos ', gameState.getPacmanPosition(), 'ghost positions ', gameState.getGhostPositions())
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(bcolors.OKBLUE, ' chosen move ', legalMoves[chosenIndex], 'best scores ', bestScore, bcolors.ENDC)
        # print('+++++++++++++++++++++++++++++++++')

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()
        score = 0  # 100*len(newGhostStates)
        for pos in newGhostPositions:
            distFromGhost = manhattan_dist(newPos, pos)
            distFromGhost +=0.01  # avoid 0 division
            # print('distFromGhost ', distFromGhost)
            if distFromGhost <=4:
                score -= 200/distFromGhost
                # break
        pq = PriorityQueue()
        for foodPos in newFood.asList():
            dist = manhattan_dist(newPos, foodPos)
            # print(bcolors.WARNING,' distFromFood ', dist, ' at', foodPos, bcolors.ENDC)
            dist += 0.01
            pq.push(dist, dist)

            # if dist<=4:
            # print(bcolors.WARNING,' distFromFood ', dist, bcolors.ENDC)
            # score += 100/dist

        while pq.isEmpty() is False:
            dist = pq.pop()
            # print(bcolors.WARNING, ' distFromFood ', dist, bcolors.ENDC)
            score += 100/dist

        # score += 500/(newFood.count()+0.1)
        if currentGameState.getFood().count()>successorGameState.getFood().count():
            # print(bcolors.BOLD, 'Will get food', bcolors.ENDC)
            score += 200

        "*** YOUR CODE HERE ***"
        # print(action, newPos, newFood.count(), newGhostPositions, newScaredTimes, ' score ', score)
        return score  # successorGameState.getScore()

def manhattan_dist(p,q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    # def minAgent(self, gameState, agentIndex, depth):
    #     if depth>self.depth:
    #         if
    #     v = float('inf')
    #     legalMoves = gameState.getLegalActions(agentIndex)
    #     nextMove = 'Stop'
    #     for move in legalMoves:
    #         nextState = gameState.generateSuccessor(agentIndex, move)
    #         # v = min(v, self.evaluationFunction(nextState))
    #         newValue = self.evaluationFunction(nextState)
    #         if newValue<v:
    #             v = newValue
    #             nextMove = move
    #     return v

    # def maxAgent(self, gameState, depth, agentIndex):
    #     v = -float('inf')
    #     legalMoves = gameState.getLegalActions(agentIndex)
    #     scores = []
    #     for move in legalMoves:
    #         nextState = gameState.generateSuccessor(agentIndex, move)
    #         for agent in range(1,gameState.getNumAgents()):
    #             score, move = minAgent(nextState, depth , agent)
    #             scores.append(score)
    #             # v = max(v, minAgent(nextState, depth , agent))
    #     bestScore = max(scores)
    #     bestIndex = [index for index in range(len(scores)) if scores[index] is bestScore]
    #     index = random.choice(bestIndex)
    #     move = legalMoves[index]
    #
    #     return bestScore, move

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # legalMoves = gameState.getLegalActions()
        # for agentIndex in gameState.getnumAgents():
        #     scores = [self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)) for action in legalMoves]
        #     bestScore = max(scores)
        #     bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #     chosenIndex = random.choice(bestIndices)
        depth = 0
        agent = 0
        score, move = self.minimaxSearch(gameState, depth, agent)

        return move
        # util.raiseNotDefined()

    def minimaxSearch(self, state, depth, agent):
        # returns [score, move]
        if depth is self.depth or state.isWin() or state.isLose():
            # print(depth, self.depth)
            return [self.evaluationFunction(state), None]

        # if agent%state.getNumAgents() is 0:
        #         #     agentType = 'MAX'
        #         # else:
        #         #     agentType = 'MIN'
        agent = agent%state.getNumAgents()

        if agent is 0:  # pacman
            # '''max agent'''
            # maxAgent(state, depth)
            v = [-float("inf"), None]
            # legalMoves = gameState.getLegalActions(0)
            # for move in legalMoves:
            #     nextState = gameState.generateSuccessor(0, move)
            #     score, move = self.minimaxSearch(state=nextState, depth=depth, agent='MIN')
            #     if score>v[0]:
            #         v = [score, move]
        else:
            # '''min agent'''
            v = [float("inf"), None]
            # minAgent(state, depth)
        # for agent in range(gameState.getNumAgents()):
        legalMoves = state.getLegalActions(agent)
        # print('agent ', agent, 'legalMoves ', legalMoves)
        if agent==state.getNumAgents()-1:
            depth+=1
        for move in legalMoves:
            nextState = state.generateSuccessor(agent, move)
            score, moveReturned = self.minimaxSearch(state=nextState, depth=depth, agent=agent+1)
            if agent is 0:
                if score>v[0]:
                    v = [score, move]
            else:
                if score<v[0]:
                    v = [score, move]
        # print('agent ', agent, 'v ', v)
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agent = 0
        alphabeta = [-float("inf")]
        for i in range(1, gameState.getNumAgents()):
            alphabeta.append(float("inf"))

        score, move = self.AlphaBetaSearch(gameState, depth, agent, alphabeta)
        return move

    def AlphaBetaSearch(self, state, depth, agent, alphabeta):
        # returns [score, move]
        if depth is self.depth or state.isWin() or state.isLose():
            # print(depth, self.depth)
            return [self.evaluationFunction(state), None]

        # if agent%state.getNumAgents() is 0:
        #         #     agentType = 'MAX'
        #         # else:
        #         #     agentType = 'MIN'
        agent = agent%state.getNumAgents()

        if agent is 0:  # pacman
            # '''max agent'''
            # maxAgent(state, depth)
            v = [-float("inf"), None]
            # legalMoves = gameState.getLegalActions(0)
            # for move in legalMoves:
            #     nextState = gameState.generateSuccessor(0, move)
            #     score, move = self.minimaxSearch(state=nextState, depth=depth, agent='MIN')
            #     if score>v[0]:
            #         v = [score, move]
        else:
            # '''min agent'''
            v = [float("inf"), None]
            # minAgent(state, depth)
        # for agent in range(gameState.getNumAgents()):
        legalMoves = state.getLegalActions(agent)  # find the moves to take
        # print('agent ', agent, 'legalMoves ', legalMoves)
        if agent==state.getNumAgents()-1:
            depth+=1
        # print(bcolors.OKBLUE, 'agent ', agent, 'legalMoves ', legalMoves, bcolors.ENDC)
        for move in legalMoves:
            # print(bcolors.FAIL, 'agent ', agent, 'move ', move, 'ENTER', bcolors.ENDC)
            nextState = state.generateSuccessor(agent, move)  # get the state after making the move
            score, moveReturned = self.AlphaBetaSearch(state=nextState, depth=depth, agent=agent+1, alphabeta=alphabeta[:])  # find the value from the new state
            # print(bcolors.FAIL, 'agent ', agent, 'move ', move, 'score ', score, 'bestScore ', v[0], 'alphabeta ', alphabeta, bcolors.ENDC)
            if agent is 0:
                if score>v[0]:  # if this state's value is greater than previous value for this turn i.e. best move
                    v = [score, move]
                if v[0]>alphabeta[0]:  # if the current value is greater than previous best solution for MAX player
                    alphabeta[0] = v[0]
                    # print('alphabeta[{0}] {1}' .format(agent, alphabeta[agent]))
                    # alphabeta[0] = max(v[0],alphabeta[0])
                if alphabeta[0]> min(alphabeta[1:]):  # if the current value is greater than MIN player's best possible value
                    # if alpha value is greater than beta value of any agent, rest of the solution isn't worth exploring
                    # print('alphabeta[0]> alphabeta[agent+1]: ', alphabeta[0], alphabeta[agent+1])
                    break

            else:
                if score<v[0]:  # if this state's value is lesser than previous value for this turn
                    v = [score, move]
                if v[0]<alphabeta[agent]:  # if the current value is lesser than previous best solution for MIN player
                    alphabeta[agent] = v[0]
                    # print('alphabeta[{0}] {1}' .format(agent, alphabeta[agent]))
                if alphabeta[0]>alphabeta[agent]:  # if the current value is lesser than MAX player's best possible value
                    # print('alphabeta[0]>alphabeta[agent]: ', alphabeta[0], alphabeta[agent])
                    break
                # alphabeta[agent] = min(v[0], alphabeta[agent])

        # print(bcolors.FAIL, 'agent ', agent, 'v ', v, 'EXIT', bcolors.ENDC)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agent = 0
        score, move = self.ExpectimaxSearch(state=gameState, depth=depth, agent=agent)
        # util.raiseNotDefined()
        return move

    def ExpectimaxSearch(self, state, depth, agent):
        # returns [score, move]
        if depth is self.depth or state.isWin() or state.isLose():
            # print(depth, self.depth)
            return [self.evaluationFunction(state), None]

        agent = agent%state.getNumAgents()

        if agent is 0:  # pacman
            # '''max agent'''
            v = [-float("inf"), None]
        else:
            # '''min agent'''
            v = [0, None]

        legalMoves = state.getLegalActions(agent)  # find the moves to take
        if agent==state.getNumAgents()-1:
            depth+=1
        # print(bcolors.OKBLUE, 'agent ', agent, 'legalMoves ', legalMoves, bcolors.ENDC)
        for move in legalMoves:
            # print(bcolors.FAIL, 'agent ', agent, 'move ', move, 'ENTER', bcolors.ENDC)
            nextState = state.generateSuccessor(agent, move)  # get the state after making the move
            score, moveReturned = self.ExpectimaxSearch(state=nextState, depth=depth, agent=agent+1)  # find the value from the new state
            # print(bcolors.FAIL, 'agent ', agent, 'move ', move, 'score ', score, 'bestScore ', v[0], bcolors.ENDC)
            if agent is 0:
                if score>v[0]:  # if this state's value is greater than previous value for this turn i.e. best move
                    v = [score, move]
                # if v[0]>alphabeta[0]:  # if the current value is greater than previous best solution for MAX player
                #     alphabeta[0] = v[0]
                #     # print('alphabeta[{0}] {1}' .format(agent, alphabeta[agent]))
                #     # alphabeta[0] = max(v[0],alphabeta[0])
                # if alphabeta[0]> min(alphabeta[1:]):  # if the current value is greater than MIN player's best possible value
                #     # if alpha value is greater than beta value of any agent, rest of the solution isn't worth exploring
                #     # print('alphabeta[0]> alphabeta[agent+1]: ', alphabeta[0], alphabeta[agent+1])
                #     break

            else:
                v[0] += score/len(legalMoves)
                # if score<v[0]:  # if this state's value is lesser than previous value for this turn
                #     v = [score, move]
                # if v[0]<alphabeta[agent]:  # if the current value is lesser than previous best solution for MIN player
                #     alphabeta[agent] = v[0]
                #     # print('alphabeta[{0}] {1}' .format(agent, alphabeta[agent]))
                # if alphabeta[0]>alphabeta[agent]:  # if the current value is lesser than MAX player's best possible value
                #     # print('alphabeta[0]>alphabeta[agent]: ', alphabeta[0], alphabeta[agent])
                #     break
                # alphabeta[agent] = min(v[0], alphabeta[agent])

        # print(bcolors.FAIL, 'agent ', agent, 'v ', v, 'EXIT', bcolors.ENDC)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # value = 0
    "*** YOUR CODE HERE ***"
    legalActions = currentGameState.getLegalActions()
    # print('legalActions ', legalActions)
    # # for action in legalActions:
    # successorGameState = currentGameState.generatePacmanSuccessor(action)  # TODO what actions to take
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPositions = successorGameState.getGhostPositions()
    newCapsules = successorGameState.getCapsules()
    # print(newGhostPositions, newScaredTimes)
    score = 0  # 100*len(newGhostStates)
    for i in range(len(newGhostPositions)):
        pos = newGhostPositions[i]
        if newScaredTimes[i] is 0:
            distFromGhost = manhattan_dist(newPos, pos)
            distFromGhost += 0.01  # avoid 0 division
            # print('distFromGhost ', distFromGhost)
            if distFromGhost <= 4:
                score -= 50 / distFromGhost
            else:
                score -= 30/distFromGhost
        # else:
            # print(bcolors.OKGREEN, 'Ghost scared!', bcolors.ENDC)
        # score += 50*newScaredTimes[i]
    # print('score += 10*(sum(newScaredTimes)>0)')
    score += 10*(sum(newScaredTimes))
            # break
    pq = PriorityQueue()
    for foodPos in newFood.asList():
        dist = manhattan_dist(newPos, foodPos)
        # print(bcolors.WARNING,' distFromFood ', dist, ' at', foodPos, bcolors.ENDC)
        dist += 0.01
        pq.push([dist, foodPos], dist)

    # if dist<=4:
    # print(bcolors.WARNING,' distFromFood ', dist, bcolors.ENDC)
    # score += 100/dist

    # while pq.isEmpty() is False:  # for all foods
    # dist = 0
    # if pq.isEmpty() is False:
    #     dist = mazeDistance(newPos, pq.pop()[1], successorGameState)
    #     # print(bcolors.WARNING, ' distFromFood ', dist, bcolors.ENDC)
    # score += 100/(dist+0.1)

    dist = 0
    while pq.isEmpty() is False:  # for nearest food
        dist += pq.pop()[0]

    score += 80 / (dist+0.1)

    # print(str(successorGameState.getCapsules()))
    # print('len(newCapsules)', len(newCapsules))
    if sum(newScaredTimes) is 0:
        dist = 0
        for capsule in newCapsules:
            dist += manhattan_dist(newPos, capsule)
            # print(bcolors.BOLD, 'dist from capsule', dist, bcolors.ENDC)
        score += 50/(dist+0.1)

    # score += 50/(len(newCapsules)+0.1)
    # score += 100/successorGameState.getCapsule().count()

    score += 700/(newFood.count()+0.1)
    # for action in legalActions:
    #     successorGameState = currentGameState.generatePacmanSuccessor(action)
    #     if currentGameState.getFood().count() > successorGameState.getFood().count():
    #         # print(bcolors.BOLD, 'Will get food', bcolors.ENDC)
    #         score += 200

    # value += score/len(legalActions)
    # print('score ', score)  # , 'value ', value)
    # score += 5*random.choice(range(2))
    score += 5*random.random()

    # print(action, newPos, newFood.count(), newGhostPositions, newScaredTimes, ' score ', score)
    return score  # successorGameState.getScore()
    # util.raiseNotDefined()

def euclidian_dist(p, q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    dist = x*x + y*y
    return dist
# Abbreviation
better = betterEvaluationFunction

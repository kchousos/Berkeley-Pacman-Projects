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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = 0

        if successorGameState.isLose():
            score = -99
            return successorGameState.getScore() + score

        if successorGameState.isWin():
            score = 99
            return successorGameState.getScore() + score

        # calculate the distance to every ghost that isn't scared
        ghostDistance = [
            manhattanDistance(ghost.getPosition(), newPos)
            for ghost in successorGameState.getGhostStates()
            if (ghost.scaredTimer == 0)
        ]

        # if there are ghosts that are not scared, decrease the score
        # by the inverse of the distance to the closest one
        if len(ghostDistance):
            nearestGhost = min(ghostDistance)
            score -= 1 / nearestGhost

        # add to the score the inverse of the distance to the closest food
        foodDistance = [manhattanDistance(food, newPos) for food in newFood.asList()]
        nearestFood = min(foodDistance)
        score += 1 / nearestFood

        return successorGameState.getScore() + score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        def terminal(state: gameState, depth):

            return state.isWin() or state.isLose() or depth == self.depth

        def maxValue(state: GameState, depth):

            legalActions = state.getLegalActions(0)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = -float("inf")

            for action in legalActions:
                v = max(v, minValue(state.generateSuccessor(0, action), 1, depth))

            return v

        def minValue(state: GameState, agent, depth):

            legalActions = state.getLegalActions(agent)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = float("inf")

            # if the next agent is pacman, then call maxvalue
            # and increase the depth by 1
            if agent == gameState.getNumAgents() - 1:
                for action in legalActions:
                    v = min(
                        v, maxValue(state.generateSuccessor(agent, action), depth + 1)
                    )

                return v

            # else if the next agent is another ghost,
            # call minvalue for the same depth
            for action in legalActions:
                v = min(
                    v,
                    minValue(state.generateSuccessor(agent, action), agent + 1, depth),
                )

            return v

        legalActions = gameState.getLegalActions(0)

        # create a dictionary with 'action':'value' pairs
        actions = {}
        for action in legalActions:
            actions[action] = minValue(gameState.generateSuccessor(0, action), 1, 0)

        # return the action with the highest value
        return max(actions, key=actions.get)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def terminal(state: gameState, depth):

            return state.isWin() or state.isLose() or depth == self.depth

        def maxValue(state: GameState, depth, alpha, beta):

            legalActions = state.getLegalActions(0)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = -float("inf")

            for action in legalActions:
                v = max(
                    v,
                    minValue(state.generateSuccessor(0, action), 1, depth, alpha, beta),
                )
                if v > beta:
                    break
                alpha = max(alpha, v)

            return v

        def minValue(state: GameState, agent, depth, alpha, beta):

            legalActions = state.getLegalActions(agent)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = float("inf")

            if agent == gameState.getNumAgents() - 1:
                for action in legalActions:
                    v = min(
                        v,
                        maxValue(
                            state.generateSuccessor(agent, action),
                            depth + 1,
                            alpha,
                            beta,
                        ),
                    )
                    if v < alpha:
                        break
                    beta = min(beta, v)

                return v

            for action in legalActions:
                v = min(
                    v,
                    minValue(
                        state.generateSuccessor(agent, action),
                        agent + 1,
                        depth,
                        alpha,
                        beta,
                    ),
                )
                if v < alpha:
                    break
                beta = min(beta, v)

            return v

        legalActions = gameState.getLegalActions(0)

        alpha = -float("inf")
        beta = float("inf")
        # create a dictionary with 'action':'value' pairs
        actions = {}
        for action in legalActions:
            actions[action] = minValue(
                gameState.generateSuccessor(0, action), 1, 0, alpha, beta
            )

            if actions[action] > beta:
                return action
            alpha = max(actions[action], alpha)

        # return the action with the highest value
        return max(actions, key=actions.get)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def terminal(state: gameState, depth):

            return state.isWin() or state.isLose() or depth == self.depth

        def maxValue(state: GameState, depth):

            legalActions = state.getLegalActions(0)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = -float("inf")

            for action in legalActions:
                v = max(v, expValue(state.generateSuccessor(0, action), 1, depth))

            return v

        def expValue(state: GameState, agent, depth):

            legalActions = state.getLegalActions(agent)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = 0

            for action in legalActions:
                # if the next agent is pacman, then call maxvalue
                # and increase the depth by 1
                if agent == gameState.getNumAgents() - 1:
                    v2 = maxValue(state.generateSuccessor(agent, action), depth + 1)

                else:
                    v2 = expValue(
                        state.generateSuccessor(agent, action), agent + 1, depth
                    )

                # add to v the probability of v2
                v += v2 / len(legalActions)

            return v

        legalActions = gameState.getLegalActions(0)

        # create a dictionary with 'action':'value' pairs
        actions = {}
        for action in legalActions:
            actions[action] = expValue(gameState.generateSuccessor(0, action), 1, 0)

        # return the action with the highest value
        return max(actions, key=actions.get)


def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: The function checks if the state is a winning or a losing
    one. If not, then it subtracts from the score if there are ghosts
    dangerously close and it adds to the score based on how near there
    is food or a capsule.
    """

    currPos = state.getPacmanPosition()
    currFoodList = state.getFood().asList()
    currGhostStates = state.getGhostStates()
    currCapsules = state.getCapsules()

    # worst case
    if state.isLose():
        score = -99
        return state.getScore() + score

    # best case
    if state.isWin():
        score = 99
        return state.getScore() + score

    score = 0

    # ghosts that aren't scared, meaning they're a threat
    ghostDistance = [
        util.manhattanDistance(currPos, ghost.getPosition())
        for ghost in currGhostStates
        if (ghost.scaredTimer == 0)
    ]

    # if there are such ghosts:
    # if there isn't such a ghost near pacman
    # increase the score by the inverse of the closest one
    if ghostDistance:
        nearestCurrentGhost = min(ghostDistance)
        if nearestCurrentGhost >= 1:
            score += 1 / nearestCurrentGhost

    # the nearest there is food, the better
    foodDistance = [util.manhattanDistance(currPos, food) for food in currFoodList]
    nearestFood = min(foodDistance)
    score += 1 / nearestFood

    # if there are capsules, the nearer the better
    if currCapsules:
        capsuleDistance = [
            util.manhattanDistance(currPos, capsule) for capsule in currCapsules
        ]
        nearestCapsule = min(capsuleDistance)
        score += 1 / nearestCapsule

    return state.getScore() + score

# Abbreviation
better = betterEvaluationFunction

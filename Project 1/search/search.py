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


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    frontier = util.Stack()
    expanded = set()
    path = []
    startNode = problem.getStartState()

    # The frontier starts empty, and we firstly push the startNode
    frontier.push((startNode, path))

    while not frontier.isEmpty():
        # Take the node that the frontier pops.
        #
        # In this case the frontier functions as a stack,
        # meaning that the order of the nodes is in LIFO
        # order. Hence, the frontier will pop the last
        # node that was inserted.
        node, path = frontier.pop()

        if problem.isGoalState(node):
            return path

        # If we haven't visited and expanded this node yet
        # we push into the frontier its child nodes, after
        # we add it to the expanded set.
        #
        # The path of any child is the path of its parent
        # plus the action to get from parent to child.
        if node not in expanded:
            expanded.add(node)
            for childNode, childPath, childCost in problem.getSuccessors(node):
                newPath = path + [childPath]
                frontier.push((childNode, newPath))


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    # This time the frontier is a Queue (FIFO)
    frontier = util.Queue()
    expanded = set()
    path = []
    startNode = problem.getStartState()

    frontier.push((startNode, path))

    while not frontier.isEmpty():
        # Now the frontier will pop the first node
        # that was inserted, since it now follows
        # a FIFO order.
        node, path = frontier.pop()

        if problem.isGoalState(node):
            return path

        # We use the same procedure as in DFS
        if node not in expanded:
            expanded.add(node)
            for childNode, childPath, childCost in problem.getSuccessors(node):
                newPath = path + [childPath]
                frontier.push((childNode, newPath))


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""

    # Now the frontier is a Priority Queue
    frontier = util.PriorityQueue()
    expanded = set()
    path = []
    startNode = problem.getStartState()

    # Now a node is not represented only by
    # its coordinates and its path, but by
    # its cost as well.
    #
    # The cost of the startNode is 0 since
    # we're already there.
    frontier.push((startNode, path, 0), 0)

    while not frontier.isEmpty():
        # The frontier will pop the node with
        # the minimum cost.
        node, path, cost = frontier.pop()

        if problem.isGoalState(node):
            return path

        # Same as before, but this time we
        # calculate the cost as well. The cost
        # to get to a childNode is the cost of its
        # parent plus the cost from parent to child,
        # similiarly to the path.
        if node not in expanded:
            expanded.add(node)
            for childNode, childPath, childCost in problem.getSuccessors(node):
                newPath = path + [childPath]
                newCost = cost + childCost
                frontier.push((childNode, newPath, newCost), newCost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    frontier = util.PriorityQueue()
    expanded = set()
    path = []
    startNode = problem.getStartState()

    frontier.push((startNode, path, 0), 0)

    while not frontier.isEmpty():
        node, path, cost = frontier.pop()

        if problem.isGoalState(node):
            return path

        # The implementation is the same as of UCS's
        # but this time we choose the optimal node
        # based on a heuristic instead of purely its
        # cost.
        #
        # The heuristic value of each node is its
        # cost plus the value of the heuristic
        # function for that node.
        if node not in expanded:
            expanded.add(node)
            for childNode, childPath, childCost in problem.getSuccessors(node):
                newPath = path + [childPath]
                newCost = cost + childCost
                newHeuristic = newCost + heuristic(childNode, problem)
                frontier.push((childNode, newPath, newCost), newHeuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

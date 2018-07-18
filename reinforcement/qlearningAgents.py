# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random,util,math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop,SGD
import numpy as np


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.QValue=util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValue[(state,action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #print("legal actions :", self.getLegalActions(state))
        if len(self.getLegalActions(state))==0:
            value = 0.0
        else:
            maxvalue=-1e10
            for action in self.getLegalActions(state):
                qvalue=self.getQValue(state, action)
                if qvalue>maxvalue:
                    maxvalue=qvalue
                    value=maxvalue
               # print("value=",value)
        return value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #print("legal actions :", self.getLegalActions(state))
        if len(self.getLegalActions(state))==0:
            return None
        else:
            qvalues=util.Counter()
            for action in self.getLegalActions(state):
                qvalues[action]=self.getQValue(state, action)
        #print("action chosen :",qvalues.argMax() )
        return qvalues.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(state))==0:
            return None
        if util.flipCoin(self.epsilon):
                action=random.choice(self.getLegalActions(state))
        else:
                action=self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        self.QValue[(state,action)]+=self.alpha*(reward+\
        self.discount*self.getValue(nextState)\
        -self.getQValue(state,action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getQValueCounter(self): #(bruno modified)
        """
          Returns QValues Counter
        """
        return self.QValue


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        
#        ReinforcementAgent.__init__(self, numTraining=args['numTraining'])
        #Load saved QValue
        if len(args.get('fileqvalues',''))!=0:
            print("Use saved QValues")
            import cPickle
            f = file(args.get('fileqvalues',''), 'r')
            self.QValue=cPickle.load(f)
            f.close()
        else:
            self.QValue=util.Counter()


    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        QValue = 0.
        features=self.featExtractor.getFeatures(state, action)
        for fi in features:
          QValue += features[fi] * self.weights[fi]
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        difference=(reward+self.discount*self.getValue(nextState)-self.getQValue(state,action))
        for fi in self.featExtractor.getFeatures(state,action):
            self.weights[fi]+=self.alpha*difference*self.featExtractor.getFeatures(state,action)[fi]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class NeuralNetQAgent(PacmanQAgent):
    from experienceReplayHelper import ExperienceReplayHelper
    from deepLearningModels import OneNeuronNN

    def __init__(self, extractor='IdentityExtractor', *args, **kwargs):
        PacmanQAgent.__init__(self, *args, **kwargs)
        self.model = None
        self.featuresExtractor = DistancesExtractor()

        #replaymemory params
        self.replayMemory = []
        self.batchSize = 32
        self.maxReplayMemorySize = 20000
        self.minReplayMemorySize = 5000

        #epsilon decreases with the number of iterations:
        self.initialEpsilon = 1
        self.finalEpsilon = 0.05
        self.epsilonSteps = 10000
        self.epsilon = self.initialEpsilon
        self.updateCount = 0

    def remember(self, state, action, reward, nextState):

        if len(self.replayMemory) > self.maxReplayMemorySize:
            self.replayMemory.pop(0)

        qState = self.featuresExtractor.getFeatures(state, action)
        nextQState = self.featuresExtractor.getFeatures(nextState, action)
        isNextStateFinal = nextState.isWin() or nextState.isLose()

        self.replayMemory.append((qState, action, reward, nextQState, isNextStateFinal))

    def sampleReplayBatch(self, size):
        return random.sample(self.replayMemory, size)

#    def getQValue(self, state, action):
#        print("in NeuralNetQAgent.getQValue")
#        if self.model is None:
#            self.initModel(state)
#        prediction = self.model.predict(state, action)
#        return prediction

#    def update(self, state, action, nextState, reward):
#        print("in NeuralNetQAgent.update")
#        if self.model is None:
#            self.initModel(state)

#        maxQ = 0
#        for a in self.getLegalActions(nextState):
#            if self.getQValue(state, action) > maxQ:
#                maxQ = self.getQValue(state, action)

#        y = reward + (self.discount * maxQ)

#        self.model.update(nextState, action, y)

    #update using replayMemory
    def update(self, state, action, nextState, reward):
        """
           Update Q-Function based on transition
        """
        if self.model is None:
            self.initModel(state)

        self.remember(state, action, util.rescale(reward, -510, 1000, -1, 1), nextState)

        if len(self.replayMemory) < 1000:#self.minReplayMemorySize:
            return #no update of parameters till enough experience gathered?

        rawBatch = self.sampleReplayBatch(self.batchSize)

        trainingBatchQStates = []        #data input for NN
        trainingBatchTargetQValues = []  #qvalue corresponding to these data

        for aQState, anAction, aReward, aNextQState, isNextStateFinal in rawBatch:

            actionsQValues = self.model.predict(np.array([aQState]))[0]

            nextActionsQValues = self.model.predict(np.array([aNextQState]))[0]
            maxNextActionQValue = max(nextActionsQValues) #max over the actions

            # Update rule
            if isNextStateFinal:
                updatedQValueForAction = aReward
            else:
                updatedQValueForAction = (aReward + self.discount * maxNextActionQValue)

            targetQValues = actionsQValues.copy()
            targetQValues[Directions.getIndex(anAction)] = updatedQValueForAction

            trainingBatchQStates.append(aQState)
            trainingBatchTargetQValues.append(targetQValues)

        self.model.train_on_batch(x=np.array(trainingBatchQStates), y=np.array(trainingBatchTargetQValues))
        self.updateCount += 1
        self.epsilon = max(self.finalEpsilon, 1.00 - float(self.updateCount) / float(self.epsilonSteps))

    def getAction(self, state):

        if self.model is None: self.initModel(state)

        # Pick Action
        legalActions = self.getLegalActions(state)
        legalActions.remove(Directions.STOP)

        if util.flipCoin(self.epsilon):
            randaction=random.choice(legalActions)
            self.doAction(state,randaction)
            return randaction

        else:
            qState = self.featuresExtractor.getFeatures(state, None)
            qValues = list(enumerate(self.model.predict(np.array([qState]))[0]))
            #sort with respect to action index
            qValues = sorted(qValues, key=lambda x: x[1], reverse=True)

            #index, element = max(enumerate(qValues), key=itemgetter(1))

            for index, qValue in qValues:
                action = Directions.fromIndex(index)
                if action in legalActions:
                    self.doAction(state,action)
                    return action


    def initModel(self, sampleState):

        qState = self.featuresExtractor.getFeatures(sampleState, Directions.NORTH)

        inputDimensions = len(qState)
#        OneNeuronNN(inputDimensions=inputDimensions,learningRate=0.01,activation='linear')
        self.activation = "linear"
        self.learningRate = 0.01

        self.model = Sequential()
        self.model.add(Dense(output_dim=4, input_dim=inputDimensions,
                                    activation=self.activation, init='uniform'))

        optimizer = SGD(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

#    def final(self, state):
#        "Called at the end of each game."
#        # call the super-class final method
#        PacmanQAgent.final(self, state)

#        # did we finish training?
#        if self.episodesSoFar == self.numTraining:
#            # you might want to print your weights here for debugging
#            "*** YOUR CODE HERE ***"
#            pass
#class NeuralNetwork:
#    def __init__(self, state):
##        walls = state.getWalls()
##        self.width = walls.width
##        print("self.width",self.width)
##        self.height = walls.height
##        print("self.height",self.height)
##        self.size = 5 * self.width * self.height
#        qState = self.featuresExtractor.getFeatures(state, Directions.NORTH)
#        inputDimensions = len(qState)
#        #OneNeuronNN()
#        self.learningRate = 0.01
#        self.activation='linear'
#        self.model = Sequential()
#        self.model.add(layers.Dense(output_dim=4, input_dim=inputDimensions,
#                                    activation=self.activation, init='uniform'))
#        optimizer = keras.optimizers.SGD(lr=self.learningRate)
#        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

#        rms = RMSprop()
#        self.model.compile(loss='mse', optimizer=rms)

#    def predict(self, state, action):
#        reshaped_state = self.reshape(state, action)
#        return self.model.predict(reshaped_state, batch_size=1)[0][0]

#    def update(self, state, action, y):
#        reshaped_state = self.reshape(state, action)
#        y = [[y]]
#        self.model.fit(reshaped_state, y, batch_size=1, epochs=1, verbose=1)

#    def reshape(self, state, action):
#        qState = np.array(SimpleExtractor().getFeatures(state, action).values()).astype(dtype=float)
#        return reshaped_state
#        reshaped_state = np.empty((1, 2 * self.size))
#        food = state.getFood()
#        walls = state.getWalls()
#        for x in range(self.width):
#            for y in range(self.height):
#                reshaped_state[0][x * self.width + y] = int(food[x][y])
#                reshaped_state[0][self.size + x * self.width + y] = int(walls[x][y])
#        ghosts = state.getGhostPositions()
#        ghost_states = np.zeros((1, self.size))
#        for g in ghosts:
#            ghost_states[0][int(g[0] * self.width + g[1])] = int(1)
#        x, y = state.getPacmanPosition()
#        dx, dy = Actions.directionToVector(action)
#        next_x, next_y = int(x + dx), int(y + dy)
#        pacman_state = np.zeros((1, self.size))
#        pacman_state[0][int(x * self.width + y)] = 1
#        pacman_nextState = np.zeros((1, self.size))
#        pacman_nextState[0][int(next_x * self.width + next_y)] = 1
#        reshaped_state = np.concatenate((reshaped_state, ghost_states, pacman_state, pacman_nextState), axis=1)
#        print("ghost_state (type,shape)",type(ghost_states),ghost_states.shape)
#        print("pacman_state (type,shape)",type(pacman_state),pacman_state.shape)
#        print("type(reshaped_state),reshaped_state.shape",type(reshaped_state),reshaped_state.shape)
#        print("reshaped_state[0][self.size:2*self.size]",reshaped_state[0][self.size:2*self.size])
#        return reshaped_state

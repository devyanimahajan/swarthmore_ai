########################################
# CS63: Artificial Intelligence, Lab 5
# Fall 2022, Swarthmore College
########################################


from numpy import dot, exp, allclose
from numpy.random import normal, shuffle, seed

from functools import partial
from itertools import product

class Node(object):
    def __init__(self, name, learning_rate=1.0):
        self.name = name
        self.in_edges = []
        self.out_edges = []
        self.activation = 0.0 # set by self.compute_activation
        self.delta = 0.0 # set by self.compute_*_delta
        self.learning_rate = learning_rate

    def __repr__(self):
        return self.name

class InputNode(Node):
    """
    A node whose only purpose is to store an activation.

    The first layer of a neural network will consist of InputNode
    objects.  Input nodes should have no in_edges, and will never
    update self.delta.  They have no activation function, and instead
    get their activation set directly by the neural network.
    """
    def __init__(self, name):
        Node.__init__(self, name, 0.0)

class BiasNode(Node):
    """
    A node that always stores activation=1.

    A neural network has a single bias node that does not appear in
    any layer.  However, the bias node is connected to all hidden and
    output nodes, and the weights on the edges from bias to those
    nodes will be updated during backpropagation.
    """
    def __init__(self):
        Node.__init__(self, "bias")
        self.activation = 1.0

class SigmoidNode(Node):
    """
    Neuron with a sigmoid activation function.

    All methods update internal state, and therefore have no return
    value.
    """
    def compute_activation(self):
        """
        Assumes that activations for nodes in the previous layer are
        already up to date.

        Computes 1 / (1 + e^-x) where x is the weighted sum of inputs.

        Stores the result in self.activation.
        """
        weighted_sum = 0
        for edge in self.in_edges:
            weighted_sum += edge.weight * edge.source.activation
        self.activation = 1/(1 + exp(-weighted_sum))

    def compute_output_delta(self, target):
        """
        This is called on output nodes in backpropagation.

        Computes out * (1 - out) * (target - out).

        The out above represents the node's current activation.
        Stores the result in self.delta.
        """
        # output is node's activation
        self.delta = self.activation * (1 - self.activation) * (target - self.activation)

    def compute_hidden_delta(self):
        """
        Assumes that the deltas for nodes in the next layer are
        already up to date. This is called on hidden nodes
        in  backpropagation.

        Computes the weighted sum of next-layer deltas.

        BE SURE TO USE EACH NODE'S old_weight WHEN CALCULATING DELTA

        Multiplies this weighted sum times the derivative of the
        activation funtion.

        (out)(1 - out)(weighted sum of next-layer deltas)

        Stores the result in self.delta.
        """
        weighted_sum_deltas = 0
        for edge in self.out_edges:
            weighted_sum_deltas += edge.old_weight * edge.dest.delta
        self.delta = self.activation * (1 - self.activation) * weighted_sum_deltas

    def update_weights(self):
        """
        Assumes that either compute_output_delta or
        compute_hidden_delta was just called to set self.delta.

        Updates the weight for each incoming edge of the node.

        Computes the new weight based on the learning_rate, delta, and
        incoming activation and then calls change_weight on each
        in-edge.

        weight += (learning_rate)(delta)(input activation)

        Use the change_weight method to modify the weight.
        """
        for edge in self.in_edges:
            weight_change = self.learning_rate * (self.delta) * edge.source.activation
            edge.change_weight(edge.weight + weight_change)

class Edge(object):
    """
    Represents a weighted edge in a neural network.

    Each edge has a source, a destination, and a weight. Edges also
    remember their most recent previous weight for computing hidden
    node deltas.
    """
    def __init__(self, source, dest, weight_func):
        """
        Initialize an edge with a random weight.

        weight_func should be a 0-argument function that returns an
        initial weight. In general, this should be a random function
        so that networks are initialized with random weights.
        """
        self.source = source
        self.dest = dest
        self.weight = weight_func()
        self.old_weight = 0.0

    def change_weight(self, new_weight):
        """
        Updates self.weight and self.old_weight.

        self.old_weight is needed for computing hidden node deltas
        during backpropagation.
        """
        self.old_weight = self.weight
        self.weight = new_weight

    def __repr__(self):
        s = "(" + self.source.name + ", "
        s += self.dest.name + ", " + str(self.weight) + ")"
        return s

class Network(object):
    """Represents a densely connected feed-forward neural network."""
    def __init__(self, input_size, output_size, hidden_sizes=[],
                 learning_rate=1.0, weight_scale=0.1, converge=0.1,
                 random_seed=None):
        """Initializes a dense multi-layer neural network.

        input_size: number of input nodes
        output_size: numer of output nodes
        hidden_sizes: list with number of hidden nodes in each hidden layer
        learning_rate: passed along to the sigmoid hidden and output nodes
        weight_scale: 'scale' parameter of the numpy.random.normal function
                  to be used when generating initial weights.
        converge: threshold for considering the network's outputs correct.
                  the train function will stop before the maximum number of
                  iterations if all outputs are within converge of their
                  targets.
        random_seed: for reproducability, seed for the pseudorandom number
                  generator can be specified. If random_seed=None, the
                  numpy seed() function will not be called. If seed is not
                  none, it will be passed to the numpy seed() function
                  before initializing edge weights.

        self.layers will be a list of lists of nodes. Each inner list
        corresponds to one layer of the neural network. The first list
        is the input layer, consisting of InputNode objects. The last
        list is the output layer; in between are hidden layers. All
        nodes in hidden and output layers are SigmoidNode objects.

        Edges are not stored directly by the network, but rather in
        the edge lists for each node. Layers are densely connected, so
        that all nodes in layer i are connected to all nodes in layer
        i+1.
        """
        if random_seed is not None:
            seed(random_seed)

        weight_func = partial(normal, 0, weight_scale)
        self.converge = converge
        self.layers = []
        self.layers.append([InputNode("in_" + str(i)) for i in range(input_size)])
        for i,layer_size in enumerate(hidden_sizes):
            self.layers.append([SigmoidNode("hidden_"+str(i)+"-"+str(j),
                                learning_rate) for j in range(layer_size)])
        self.layers.append([SigmoidNode("out_"+str(i), learning_rate) for i
                            in range(output_size)])

        # densely connect consecutive layers
        for source_layer, dest_layer in zip(self.layers, self.layers[1:]):
            for source, dest in product(source_layer, dest_layer):
                edge = Edge(source, dest, weight_func)
                source.out_edges.append(edge)
                dest.in_edges.append(edge)

        # connect each node to bias
        self.bias = BiasNode()
        for layer in self.layers[1:]:
            for node in layer:
                e = Edge(self.bias, node, weight_func)
                node.in_edges.append(e)
                self.bias.out_edges.append(e)

    def predict(self, input_vector):
        """
        Computes the network's output for a given input_vector.

        input_vector: activation value for each input node
        returns: a vector of activation values for each output node

        Sets the activation of each node in the input layer to the
        appropriate value from the input vector. Then for each
        subsequent layer, calls each node's compute_activation
        function to update its activation. Collects the activation
        values for each output node into the return vector.
        """
        #raise NotImplementedError("TODO")
        act_values = []
        #print("length of vec:",len(input_vector))
        for i in range(len(self.layers[0])):
            self.layers[0][i].activation = input_vector[i]

        for layer in self.layers[1:]:
            for node in layer:
                node.compute_activation()

        for node in self.layers[-1]:
            act_values.append(node.activation)

        return act_values

    def backpropagation(self, target_vector):
        """
        Updates all weights for a single step of stochastic gradient
        descent.

        Assumes that predict has just been called on the input vector
        corresponding to the given target_vector.

        target_vector: expected activation for each output
        returns: nothing

        Calls compute_output_delta on each node in the output layer
        and updates the weights for each output node.

        Then working BACKWARDS, calls compute_hidden_delta on all
        nodes in each hidden layer and updates the weights for each
        hidden node.
        """
        #raise NotImplementedError("TODO")

        #for (node, target) in (self.layers[-1], target_vector):
        #    node.compute_output_delta(target)
        """
        for node in self.layers[-1]:
            for target in target_vector:
                node.compute_output_delta(target)
        """
        out_layer = self.layers[-1]
        for i in range(len(out_layer)):
            out_layer[i].compute_output_delta(target_vector[i])
            out_layer[i].update_weights()

        reverse_layers = list(reversed(self.layers))
        for layer in reverse_layers[1:-1]:
            for node in layer:
                node.compute_hidden_delta()
                node.update_weights()

    def test(self, data_set):
        """
        Predicts every input in the data set and returns the accuracy.

        data_set: a list of (input_vector, target_vector) pairs
        returns: accuracy, the fraction of output vectors within self.converge
                 of their targets (as measured by numpy.allclose's atol).

        Calls predict() on each input vector in the data set, and
        compares the result to the corresponding target vector from
        the data set.
        """
        #raise NotImplementedError("TODO")
        correct = 0
        count = len(data_set)
        #print(data_set)
        for pair in data_set:
            input_vector = pair[0]
            target = pair[1]
            pred = self.predict(input_vector)
            result = allclose(pred, target, self.converge)
            if result == True:
                correct += 1
        return correct/count



    def train(self, data_set, epochs, verbose=0):
        """
        Runs repeated prediction and backpropagation steps to learn
        the data.

        data_set: a list of (input_vector, target_vector) pairs
        epochs: maximum number of times to loop through the data set.
        verbose:
          if 0, nothing is printed;
          if 1, prints the number of the epoch on which training ended;
          if 2, also prints epoch number and accuracy at the end of epoch;
        returns: nothing

        Runs epochs loops through the data set (shuffled each
        time).  Each loop runs predict to compute activations, then
        backpropagation to update weights on every example in the data
        set. If self.test returns 1.0, meaning that all outputs are
        close enough to targets, then training stops regardless of the
        the current epoch number.
        """
        epochnum = 0
        for epoch in range(epochs):
            #method 1: iterates through all epochs
            #"""
            shuffle(data_set)
            for pair in data_set:
                input_vector = pair[0]
                target = pair[1]
                pred = self.predict(input_vector)
                self.backpropagation(target)
            testval = self.test(data_set)
            epochnum += 1
            if testval == 1.0:
                break
            #"""
            #method 2: same results
            #updates new dataset with predicted values
            #maintains length to avoid index errors
            """
            shuffle(data_set)
            newdata = data_set
            for pair in newdata:
                input_vector = pair[0]
                target = pair[1]
                pred = self.predict(input_vector)
                self.backpropagation(target)
                pair = (pred,target)
            testval = self.test(newdata)
            epochnum += 1
            if testval == 1.0:
                break
            """
            #method 3: index error, probably bc lst isn't long enough
            #abandon all hope ye who enter here
            """
            shuffle(data_set)
            lst_data = []
            for pair in data_set:
                input_vector = pair[0]
                target = pair[1]
                pred = self.predict(input_vector)
                self.backpropagation(target)
                lst_data.append((pred,target))
            testval = self.test(lst_data)
            epochnum += 1
            if testval == 1.0:
                break
            """
        if verbose == 1:
            print("Epoch number:", epochnum)
        elif verbose == 2:
            print("Epoch number:", epochnum)
            print("Accuracy:",testval)




    def __repr__(self):
        s = "Neural Network\n"
        for layer in self.layers:
            s += repr(layer) + "\n"
            for node in layer:
                if node.out_edges:
                    s += "  " + repr(node.out_edges) + "\n"
        s += "  " + repr(self.bias.out_edges) + "\n"
        return s

def main():
    # train a network with one hidden layer on XOR
    data_set = [([0,0],[0.0]), ([0,1],[1.0]), ([1,0],[1.0]), ([1,1],[0.0])]
    # network has 2 input nodes, 1 output node, and a single hidden layer with 2 nodes
    nn = Network(2, 1, [2])
    nn.train(data_set, 10000, verbose=2)

    # print the resulting network
    print(nn)

    # for each input in the data set show the actual output produced
    for input_vector, target_vector in data_set:
        output_vector = nn.predict(input_vector)
        print("input:", input_vector, "target:", target_vector,\
              "output:", output_vector)

if __name__ == "__main__":
    main()

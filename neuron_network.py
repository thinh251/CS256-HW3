import numpy as np
import tensorflow as tf
import util


class NeuronNetwork(object):

    def __init__(self, x, bias, epsilon):
        super(NeuronNetwork, self).__init__()
        # self.input = tf.placeholder(tf.float32, [None, len(x)])
        self.input = x
        self.learning_rate = epsilon
        self.weights = tf.random_normal(np.shape(x[1]), stddev=0.1)
        # How to calculate bias will be discussed in class on Monday
        self.biases
        self.num_labels = 6
        # The real output from model data
        self.y_known = tf.placeholder(tf.float32, [None, self.num_labels])
        # TODO:
        # weight generate random? , how to pick b will be discussed in
        # class on Monday
        self.__define_weight_bias([5, 4, 3, 2, 1])
        # Neuron network output
        self.nn_output = self.__build_network(self.input, self.weights,
                                              self.bias)


    def __define_weight_bias(self, hidden_dimensions):
        """Define the weights and bias of hidden layers
            hidden_dimenssions: is an array whose elements is the number
            of weight element of a particular layer
            For example: [3, 2, 1] means
            - Hidden layer 1: has 3 weight
            - Hidden layer 2: has 2 weight
            - Hidden layer 1: has 1 weight
        """
        self.weights = {
            'hidden_1': tf.Variable(tf.truncated_normal([len(self.input), hidden_dimensions[0]])),
            'hidden_2': tf.Variable(tf.truncated_normal([hidden_dimensions[0], hidden_dimensions[1]])),
            'hidden_3': tf.Variable(tf.truncated_normal([hidden_dimensions[1], hidden_dimensions[2]])),
            'hidden_4': tf.Variable(tf.truncated_normal([hidden_dimensions[2], hidden_dimensions[3]])),
            'hidden_5': tf.Variable(tf.truncated_normal([hidden_dimensions[3], hidden_dimensions[4]])),
            # because we have 6 sticky label
            'out': tf.Variable(tf.truncated_normal([hidden_dimensions[4], self.num_labels]))
        }

        self.bias = {
            'bias_1': tf.Variable(tf.truncated_normal(hidden_dimensions[0])),
            'bias_2': tf.Variable(tf.truncated_normal(hidden_dimensions[1])),
            'bias_3': tf.Variable(tf.truncated_normal(hidden_dimensions[2])),
            'bias_4': tf.Variable(tf.truncated_normal(hidden_dimensions[3])),
            'bias_5': tf.Variable(tf.truncated_normal(hidden_dimensions[4])),
            # because we have 6 sticky label
            'out': tf.Variable(tf.truncated_normal(self.num_labels))
        }


    def __build_network(self, x, w, b):
        """ Define 5 layers for this homework.
            x is inputs, w: weight, b: bias
        """
        layer_1 = tf.add(tf.matmul(x, w['hidden_1'], b['bias_1']))
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, w['hidden_2'], b['bias_2']))
        layer_2 = tf.nn.relu(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, w['hidden_3'], b['bias_3']))
        layer_3 = tf.nn.relu(layer_3)

        layer_4 = tf.add(tf.matmul(layer_3, w['hidden_4'], b['bias_4']))
        layer_4 = tf.nn.relu(layer_4)

        layer_5 = tf.add(tf.matmul(layer_4, w['hidden_5'], b['bias_5']))
        layer_5 = tf.nn.softmax(layer_5)

        output_layer = tf.matmul(layer_5, w['out'], b['out'])
        return output_layer


    def start(self, mode, model_file, data_folder):
        x, y = util.read_data(data_folder)
        if mode == 'train':
            # TODO:
            # Assign random weight and bias for train only mode??
            self.train(x, y, model_file)
        elif mode == '5fold':
            self.kfold(x, y, 5, model_file)
        elif mode == 'test':
            self.test(x, y, model_file)


    def train(self, train_x, train_y, model_file):
        x = tf.placeholder(tf.float32, [None, len(self.input)])
        W = tf.Variable(tf.zeros([len(self.input)], self.num_labels))
        b = tf.Variable(tf.zeros(self.num_labels))
        # y_predict = tf.placeholder(tf.float32, [None, self.num_labels])

        # Forward propagation
        # nn_output = self.define_layers(self.input, self.weights, self.biases)
        # nn_output is define in __init__ method
        predict = tf.argmax(self.nn_output, axis=1)

        # Backward propagation: update weights to minimize the cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.nn_output, labels=self.y_known))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        # mse, accuracy
        with tf.Session as session:
            session.run(init)
            for epoch in range(len(train_x)):

                session.run(optimizer, feed_dict={x: train_x, self.y_known: train_y})
                cost = session.run(cost, feed_dict={x: train_x, self.y_known: train_y})
                # y_predict = session.run(self.nn_output, feed_dict=)
        # TODO: add detail to util.write_weights_file
        util.write_weights_file(model_file, W)

    def kfold(self, x, y, k, model_file):
        # TODO
        """Split the input data into k parts and do cross-validation for these
        part
        """
        # Can we use scipy library or we have to split data set manually?
        xi = np.hsplit(x, k)
        yi = np.hsplit(y, k)

        for i in range(len(xi)):
            test_x = xi[i]
            test_y = yi[i]

    def test(self, x, y, model_file):
        #TODO
        w = util.read_weights_from(model_file)
        if None != w:
            self.weights['hidden_1'].assign(w[0])
            self.weights['hidden_2'].assign(w[1])
            self.weights['hidden_3'].assign(w[2])
            self.weights['hidden_4'].assign(w[3])
            self.weights['hidden_5'].assign(w[4])
            self.test(x, y)
        else:
            raise IOError('Weight can not be loaded from model file')

    def read_data(self, model_file):
        #  TODO: This function should parse model data file and return the
        # the input and outputs
        x, y = [], []
        return x, y

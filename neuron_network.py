import numpy as np
import tensorflow as tf
import util


class NeuronNetwork(object):

    def __init__(self, x, y, epsilon):
        super(NeuronNetwork, self).__init__()
        # self.features = tf.placeholder(tf.float32, [None, len(x)])
        self._il_node_num = 8
        self._hl1_node_num = 6
        self._hl2_node_num = 4
        self._hl3_node_num = 2
        self._ol_node_num = 1
        self._batch_size = 1000
        # self._epoch = 1000
        self.features = x # Concrete input
        self.output = y   # Concrete output
        self.learning_rate = epsilon

        # The real output from model data
        self.y_known = tf.placeholder(tf.float32, [None, self.label_count])
        self.__define_weight_bias()
        # Neuron network output
        self.nn_output = self.__build_network()

    def __define_weight_bias(self):
        """Define the weights and bias of layers
            weights are matrix hold random number between -b and b
            For now, b belongs to a set [-1, 1]
        """
        # Weights of the network is a matrix of 4 x max number of node in a layer
        self.weights = {
            'features': tf.Variable(tf.truncated_normal(shape=(len(self.features[0]), self.il_node_num), mean=0.0, stddev=1)),
            'hidden_1': tf.Variable(tf.truncated_normal(shape=(self.il_node_num, self.hl1_node_num), mean=0.0, stddev=1)),
            'hidden_2': tf.Variable(tf.truncated_normal(shape=(self.hl1_node_num, self.hl2_node_num), mean=0.0, stddev=1)),
            'hidden_3': tf.Variable(tf.truncated_normal(shape=(self.hl2_node_num, self.hl3_node_num), mean=0.0, stddev=1)),
            # because we have 6 sticky label
            'out': tf.Variable(tf.truncated_normal(shape=(self.hl3_node_num, self.ol_node_num), mean=0.0, stddev=1))
        }

        self.bias = {
            'features': tf.Variable(tf.truncated_normal(shape=(1, self.il_node_num), mean=0.0, stddev=1)),
            'bias_1': tf.Variable(tf.truncated_normal(shape=(1, self.hl1_node_num), mean=0.0, stddev=1)),
            'bias_2': tf.Variable(tf.truncated_normal(shape=(1, self.hl2_node_num), mean=0.0, stddev=1)),
            'bias_3': tf.Variable(tf.truncated_normal(shape=(1, self.hl3_node_num), mean=0.0, stddev=1)),
            # because we have 6 sticky label
            'out': tf.Variable(tf.truncated_normal(shape=(1, self.label_count), mean=0.0, stddev=1))
        }

    def __build_network(self):
        """ Define 5 layers for this homework.
            x is inputs, w: weight, b: bias
        """
        self.input = tf.placeholder(tf.float32, [None, len(self.features[0])])
        input_layer = tf.add(tf.matmul(self.input, self.weights['features']), self.bias['features'])
        input_layer = tf.nn.relu(input_layer)

        hidden_layer_1 = tf.add(tf.matmul(input_layer, self.weights['hidden_1']), self.bias['bias_1'])
        hidden_layer_1 = tf.nn.relu(hidden_layer_1)

        hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, self.weights['hidden_2']), self.bias['bias_2'])
        hidden_layer_2 = tf.nn.relu(hidden_layer_2)

        hidden_layer_3 = tf.add(tf.matmul(hidden_layer_2, self.weights['hidden_3']), self.bias['bias_3'])
        hidden_layer_3 = tf.nn.relu(hidden_layer_3)

        output_layer = tf.matmul(hidden_layer_3, self.weights['out']) + self.bias['out']
        return output_layer

    #def start(self, mode, model_file, data_folder):
    def start(self, mode, model_file):
        # x, y = util.read_data(data_folder)
        if mode == 'train':
            # TODO:
            # Assign random weight and bias for train only mode??
            self.train(self.features, self.output, model_file)
        # elif mode == '5fold':
            # self.kfold(x, y, 5, model_file)
        # elif mode == 'test':
            # self.test(x, y, model_file)

    def train(self, train_x, train_y, model_file):

        # W = tf.Variable(tf.zeros([len(self.features)], self.label_count))
        # b = tf.Variable(tf.zeros(self.label_count))

        # Evaluate model
        prediction = tf.nn.softmax(self.nn_output)

        # Backward propagation: update weights to minimize the cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.nn_output, labels=self.y_known))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        # mse, accuracy
        b = 0
        # with tf.Session as session:
        session = tf.Session()
        session.run(init)
        # batch_x, batch_y = tf.train.batch(train_x, train_y, self.batch_size)
        session.run(optimizer, feed_dict={self.input: train_x, self.y_known: train_y})
        session.run(cost, feed_dict={self.input: train_x, self.y_known: train_y})
        correct_pred = tf.equal(tf.argmax(prediction, 1),
                                tf.argmax(self.y_known, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy = session.run(accuracy, feed_dict={self.input: train_x, self.y_known: train_y})
        print b + len(train_x), ' items trained, - Accuracy: ', accuracy

        session.close()

        # TODO: add detail to util.write_weights_file
        # util.write_weights_file(model_file, W)

    def kfold(self, x, y, k, model_file):
        # TODO
        """Split the features data into k parts and do cross-validation for these
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
        if w is not None:
            self.weights['hidden_1'].assign(w[0])
            self.weights['hidden_2'].assign(w[1])
            self.weights['hidden_3'].assign(w[2])
            self.weights['hidden_4'].assign(w[3])
            self.weights['hidden_5'].assign(w[4])

            predict = tf.argmax(self.nn_output, axis=1)
            correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            init = tf.global_variables_initializer()
            with tf.Session as session:
                session.run(init)
                session.run(accuracy, feed_dict={x: x, y: y})

        else:
            raise IOError('Weight can not be loaded from model file')

    @property
    def il_node_num(self):
        return self._il_node_num

    @il_node_num.setter
    def il_node_num(self, value):
        if value <= 0:
            raise ValueError('Input layer: Number of node must greater than 0')
        else:
            self._il_node_num = value

    @property
    def hl1_node_num(self):
        return self._hl1_node_num

    @hl1_node_num.setter
    def hl1_node_num(self, value):
        if value <= 0:
            raise ValueError('Hidden layer 1: Number of node must greater than 0')
        else:
            self._hl1_node_num = value

    @property
    def hl2_node_num(self):
        return self._hl2_node_num

    @hl2_node_num.setter
    def hl2_node_num(self, value):
        if value <= 0:
            raise ValueError('Hidden layer 2: Number of node must greater than 0')
        else:
            self._hl2_node_num = value

    @property
    def hl3_node_num(self):
        return self._hl3_node_num

    @hl3_node_num.setter
    def hl3_node_num(self, value):
        if value <= 0:
            raise ValueError('Hidden layer 3: Number of node must greater than 0')
        else:
            self._hl3_node_num = value

    @property
    def ol_node_num(self):
        return self._ol_node_num

    @ol_node_num.setter
    def ol_node_num(self, value):
        if value <= 0:
            raise ValueError('Output layer: Number of node must greater than 0')
        else:
            self._ol_node_num = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            raise ValueError('Batch size must greater than 0')
        else:
            self._batch_size = value

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        if value <= 0:
            raise ValueError('Epoch must greater than 0')
        else:
            self._epoch = value

    @property
    def label_count(self):
        return 6

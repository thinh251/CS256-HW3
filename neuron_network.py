import numpy as np
import tensorflow as tf
import util
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuronNetwork(object):

    def __init__(self, x, y, epsilon):
        super(NeuronNetwork, self).__init__()
        # self.features = tf.placeholder(tf.float32, [None, len(x)])
        self._il_node_num = 5
        self._hl1_node_num = 4
        self._hl2_node_num = 3
        self._hl3_node_num = 2
        self._ol_node_num = 1
        self._batch_size = 1000
        # self._epoch = 1000
        self.features = x  # Concrete input
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
            'w1': tf.Variable(tf.random_uniform(shape=(len(self.features[1]), self.il_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32)),
            'w2': tf.Variable(tf.random_uniform(shape=(self.il_node_num, self.hl1_node_num),  minval=-1.0, maxval=1.0, dtype=tf.float32)),
            'w3': tf.Variable(tf.random_uniform(shape=(self.hl1_node_num, self.hl2_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32)),
            'w4': tf.Variable(tf.random_uniform(shape=(self.hl2_node_num, self.hl3_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32)),
            # because we have 6 sticky label
            'w5': tf.Variable(tf.truncated_normal(shape=(self.hl3_node_num, self.ol_node_num), mean=0.0, stddev=1))
        }

        self.bias = {
            'b1': tf.Variable(tf.random_uniform(shape=(1, self.il_node_num),  minval=-1.0, maxval=1.0, dtype=tf.float32)),
            'b2': tf.Variable(tf.random_uniform(shape=(1, self.hl1_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32)),
            'b3': tf.Variable(tf.random_uniform(shape=(1, self.hl2_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32)),
            'b4': tf.Variable(tf.random_uniform(shape=(1, self.hl3_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32)),
            # because we have 6 sticky label
            'b5': tf.Variable(tf.truncated_normal(shape=(1, self.label_count), mean=0.0, stddev=1))
        }

    def __build_network(self):
        """ Define 5 layers for this homework.
            x is inputs, w: weight, b: bias
        """
        self.input = tf.placeholder(tf.float32, [None, len(self.features[1])])
        input_layer = tf.add(tf.matmul(self.input, self.weights['w1']), self.bias['b1'])
        input_layer = tf.nn.relu(input_layer)

        hidden_layer_1 = tf.add(tf.matmul(input_layer, self.weights['w2']), self.bias['b2'])
        hidden_layer_1 = tf.nn.relu(hidden_layer_1)

        hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, self.weights['w3']), self.bias['b3'])
        hidden_layer_2 = tf.nn.relu(hidden_layer_2)

        hidden_layer_3 = tf.add(tf.matmul(hidden_layer_2, self.weights['w4']), self.bias['b4'])
        hidden_layer_3 = tf.nn.relu(hidden_layer_3)

        output_layer = tf.matmul(hidden_layer_3, self.weights['w5']) + self.bias['b5']
        return output_layer

    # def start(self, mode, model_file.txt, data_folder):
    def start(self, mode, model_file, data_folder):
        # x, y = util.read_data(data_folder)
        if mode == 'train':
            self.train(self.features, self.output, model_file)
        # elif mode == '5fold':
            # self.kfold(x, y, 5, model_file.txt)
        elif mode == 'test.txt':
            test_x, test_y = util.load_test_data(data_folder)
            self.test(test_x, test_y, model_file)

    def train(self, train_x, train_y, model_file):
        # Evaluate model
        prediction = tf.nn.softmax(self.nn_output)

        # Backward propagation: update weights to minimize the cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.nn_output, labels=self.y_known))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        # mse, accuracy
        b = 0
        # Open model file in overwrite mode
        f = open(model_file, 'w')
        # with tf.Session as session:
        session = tf.Session()
        session.run(init)
        start_time = datetime.now()
        # batch_x, batch_y = tf.train.batch(train_x, train_y, self.batch_size)
        session.run(optimizer, feed_dict={self.input: train_x, self.y_known: train_y})
        session.run(cost, feed_dict={self.input: train_x, self.y_known: train_y})
        correct_pred = tf.equal(tf.argmax(prediction, 1),
                                tf.argmax(self.y_known, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy = session.run(accuracy, feed_dict={self.input: train_x, self.y_known: train_y})
        print 'Processing completed:', b + len(train_x), 'items trained, - Accuracy: ', accuracy
        print 'Training time: ', datetime.now() - start_time
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # 5 layers
        print("Current W: ")
        # for w in self.weights:
        #     o_w = np.array(self.weights[w].eval(session=session))
        #     print w, '\n', o_w
        w_matrix = np.zeros((5, self.max_node_of_layers, len(self.features[1])), dtype=float)
        # print "Initiated w_matrix:", w_matrix
        key_list = self.weights.keys()
        key_list.sort()
        i = 0
        for w in key_list:  # 5 layers
            layer_w = np.array(self.weights[w].eval(session=session))
            layer_w_t = layer_w.T
            print 'Layer: ', w, 'Shape: ', np.shape(layer_w_t)
            d = len(self.features[1])
            if w == 'w2':
                d = self.il_node_num
            elif w == 'w3':
                d = self.hl1_node_num
            elif w == 'w4':
                d = self.hl2_node_num
            elif w == 'w5':
                d = self.hl3_node_num
            it = np.nditer(layer_w_t, flags=['f_index'])
            while not it.finished:
                r = it.index / d
                c = it.index % d
                w_matrix[i][r][c] = it[0]
                it.iternext()
            i += 1

        print w_matrix

        with open(model_file, 'w') as f:
            s = 1
            for data_slice in w_matrix:
                f.write('#w' + str(s) + '\n')
                np.savetxt(f, data_slice, fmt='%2.3f')
                s += 1
        session.close()

        # TODO: add detail to util.write_weights_file
        # util.write_weights_file(model_file.txt, W)

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
        # print 'Before assigning weight:', self.weights['w1']
        full_path = os.path.join(os.path.curdir, model_file)
        w = np.loadtxt(full_path)
        w = np.reshape(w, (5, self.max_node_of_layers, len(self.features[1])))
        # print 'Data load:', w
        w1 = np.transpose(w[0])
        w2 = np.transpose(w[1])
        w2 = np.delete(w2, np.s_[self.il_node_num:], axis=0)
        w2 = np.delete(w2, np.s_[self.hl1_node_num:], axis=1)
        w3 = np.transpose(w[2])
        w3 = np.delete(w3, np.s_[self.hl1_node_num:], axis=0)
        w3 = np.delete(w3, np.s_[self.hl2_node_num:], axis=1)
        w4 = np.transpose(w[3])
        w4 = np.delete(w4, np.s_[self.hl2_node_num:], axis=0)
        w4 = np.delete(w4, np.s_[self.hl3_node_num:], axis=1)
        w5 = np.transpose(w[4])
        w5 = np.delete(w5, np.s_[self.hl3_node_num:], axis=0)
        w5 = np.delete(w5, np.s_[self.ol_node_num:], axis=1)
        if w is not None:
            self.weights['w1'].assign(w1)
            self.weights['w2'].assign(w2)
            self.weights['w3'].assign(w3)
            self.weights['w4'].assign(w4)
            self.weights['w5'].assign(w5)
            start_time = datetime.now()
            init = tf.global_variables_initializer()

            correct_pred = tf.equal(tf.argmax(self.nn_output, 1), tf.argmax(self.y_known, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            session = tf.Session()
            # with tf.Session as session:
            session.run(init)
            predict_y = session.run(self.nn_output, feed_dict={self.input: x, self.y_known: y})
            mse = tf.reduce_mean(tf.square(predict_y - y))
            mse = session.run(mse)
            print 'Testing completed!'
            print 'MSE: ', mse
            print 'Accuracy:', session.run(accuracy, feed_dict={self.input: x, self.y_known: y})
            session.close()
            print 'Testing time: ', datetime.now() - start_time

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
    def max_node_of_layers(self):
        return max(self.il_node_num, self.hl1_node_num, self.hl2_node_num, self.hl3_node_num, self.ol_node_num)
    # @property
    # def epoch(self):
    #     return self._epoch
    #
    # @epoch.setter
    # def epoch(self, value):
    #     if value <= 0:
    #         raise ValueError('Epoch must greater than 0')
    #     else:
    #         self._epoch = value

    @property
    def label_count(self):
        return 6

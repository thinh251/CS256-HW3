import numpy as np
import tensorflow as tf
import util
from datetime import datetime
from datetime import timedelta
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuronNetwork(object):

    def __init__(self, x, y, epsilon):
        super(NeuronNetwork, self).__init__()
        # self.input_texts = tf.placeholder(tf.float32, [None, len(x)])
        self.__il_node_num = 5
        self.__hl1_node_num = 4
        self.__hl2_node_num = 3
        self.__hl3_node_num = 2
        self.__ol_node_num = 1
        self.__mini_batch_size = 1000
        self.__epoch = 1000
        self.__kfold = 5
        self.input_data = x  # Concrete input_holder data
        self.output_data = y   # Concrete output data
        self.__number_of_features = len(self.input_data[1])
        # print 'Number of input_data: ', self.number_of_features
        self.learning_rate = epsilon
        # Hold the input_data for training and testing
        self.input_holder = tf.placeholder(tf.float32, [None, self.number_of_features], name='input-holder')
        # Hold the output_data for training and testing
        self.y_holder = tf.placeholder(tf.float32, [None, self.label_count], name='output-holder')
        self.__define_weight_bias()
        # Neuron network output_data
        self.nn_output = self.__build_network()
        self.test_accuracy_history = []
        self.train_accuracy_history = []
        self.items_trained = 0
        self.items_test = 0
        self.__confusion_matrix = {'TPR': 0, 'FPR': 0, 'FNR': 0, 'TNR': 0}

    def __define_weight_bias(self):
        """Define the weights and bias of layers
            weights are matrix hold random number between -b and b
            For now, b belongs to a set [-1, 1]
        """
        # Weights of the network is a matrix of 4 x max number of node in a layer
        self.weights = {
            'w1': tf.Variable(tf.random_uniform(shape=(self.number_of_features, self.il_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32), name='w1'),
            'w2': tf.Variable(tf.random_uniform(shape=(self.il_node_num, self.hl1_node_num),  minval=-1.0, maxval=1.0, dtype=tf.float32), name='w2'),
            'w3': tf.Variable(tf.random_uniform(shape=(self.hl1_node_num, self.hl2_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32), name='w3'),
            'w4': tf.Variable(tf.random_uniform(shape=(self.hl2_node_num, self.hl3_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32), name='w4'),
            # because we have 6 sticky label
            'w5': tf.Variable(tf.truncated_normal(shape=(self.hl3_node_num, self.ol_node_num), mean=0.0, stddev=1), name='w5')
        }

        self.bias = {
            'b1': tf.Variable(tf.random_uniform(shape=(1, self.il_node_num),  minval=-1.0, maxval=1.0, dtype=tf.float32), name="b1"),
            'b2': tf.Variable(tf.random_uniform(shape=(1, self.hl1_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32), name="b2"),
            'b3': tf.Variable(tf.random_uniform(shape=(1, self.hl2_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32), name="b3"),
            'b4': tf.Variable(tf.random_uniform(shape=(1, self.hl3_node_num), minval=-1.0, maxval=1.0, dtype=tf.float32), name="b4"),
            # because we have 6 sticky label
            'b5': tf.Variable(tf.truncated_normal(shape=(1, self.label_count), mean=0.0, stddev=1), name='b5')
        }

    def __build_network(self):
        """ Define 5 layers for this homework.
            x is input_texts, w: weight, b: bias
        """
        input_layer = tf.add(tf.matmul(self.input_holder, self.weights['w1']), self.bias['b1'])
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
    def start(self, mode, model_file):
        # x, y = util.read_data(data_folder)
        if mode == 'train':
            print 'Start training........'
            start_time = datetime.now()
            self.train(self.input_data, self.output_data, model_file, True)
            print 'Processing completed:\n',
            print self.items_trained, 'item(s) trained,'
            print 'Training time: ', datetime.now() - start_time
        elif mode == '5fold':
            training_time, testing_time = self.cross_validate_optimized(
                self.input_data, self.output_data, self.kfold, model_file)
            print '5fold completed!!!'
            print self.items_trained, ' item(s) trained,'
            print self.items_test, ' item(s) tested'
            # print 'Training accuracy: ', np.mean(self.train_accuracy_history)
            print 'Testing accuracy: ', np.mean(self.test_accuracy_history)
            print 'Training time: ', training_time
            print 'Testing time: ', training_time
        elif mode == 'test':
            # test_x, test_y = util.load_test_data(data_folder)
            start_time = datetime.now()
            self.test(self.input_data, self.output_data, model_file)
            print 'Testing completed!'
            print self.items_test, ' item(s) tested'
            print 'Accuracy:', self.test_accuracy_history
            print 'Testing time: ', datetime.now() - start_time

    def train(self, train_x, train_y, model_file, write_to_file):
        """Write to file flag to write the weight to model file
            for example: train only model this flag is on
            5fold mode: at the 5th dataset, this flag need to turn on to save the weight
        """
        # Evaluate model
        self.items_trained = 0
        prediction = tf.nn.softmax(self.nn_output)
        # Backward propagation: update weights to minimize the cost using
        # cross_entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.nn_output, labels=self.y_holder))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)
        # Count the number of training items
        items_count = 0
        # Mark the 1000 items step to print number of processed item
        grand = 1
        is_printed = False
        for e in range(self.epoch):
            # shuffle the data before training
            for i in range(0, len(train_x)):
                try:
                    j = random.randint(i + 1, len(train_x) - 1)
                    if i != j:
                        train_x[i], train_x[j] = train_x[j], train_x[i]
                        train_y[i], train_y[j] = train_y[j], train_y[i]
                except ValueError:
                    pass
                    # print 'End of the list when shuffling'

            # slice the training data into mini batches and train on these batches
            for k in range(0, len(train_x), self.batch_size):
                batch_x = train_x[k:k + self.batch_size]
                batch_y = train_y[k:k + self.batch_size]
                session.run(optimizer,
                            feed_dict={self.input_holder: batch_x,
                                       self.y_holder: batch_y})

                correct_pred = tf.equal(tf.argmax(prediction, 1),
                                        tf.argmax(self.y_holder, 1))
                accuracy = tf.reduce_mean(
                    tf.cast(correct_pred, tf.float32))
                accuracy = session.run(accuracy,
                                       feed_dict={
                                           self.input_holder: batch_x,
                                           self.y_holder: batch_y})

                self.items_trained += len(train_x)
                # The homework does not require to calculate the training accuracy
                # self.train_accuracy_history.append(accuracy)
                items_count += len(batch_x)
                jp = items_count / 1000
                if items_count > (grand * 1000) and not is_printed:
                    print str(items_count - (items_count % 1000)) + ' items processed'
                    is_printed = True
                    grand += 1
                elif jp >= grand:
                    is_printed = False

        # Save the optimized weights and the biases to the model file
        if write_to_file:
            print 'Save to file'
            saver = tf.train.Saver([self.weights['w1'], self.weights['w2'], self.weights['w3'], self.weights['w4'], self.weights['w5']])
            saver.save(session, model_file)
        # print 'W5-trained:\n', session.run(self.weights['w5'])
        # print 'b1-trained:\n', session.run(self.bias['b1'])
        session.close()
        self.items_trained = items_count
        # region This is our first naive approach to save model file using numpy
        # which requires to manipulate the matrices dimenssions
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # # 5 layers
        # # print("Current W: ")
        # w_matrix = np.zeros((5, self.max_node_of_layers, self.number_of_features), dtype=float)
        # # print "Initiated w_matrix:", w_matrix
        # key_list = self.weights.keys()
        # key_list.sort()
        # i = 0
        # for w in key_list:  # 5 layers
        #     layer_w = np.array(self.weights[w].eval(session=session))
        #     layer_w_t = layer_w.T
        #     # print 'Layer: ', w, 'Shape: ', np.shape(layer_w_t)
        #     d = self.number_of_features
        #     if w == 'w2':
        #         d = self.il_node_num
        #     elif w == 'w3':
        #         d = self.hl1_node_num
        #     elif w == 'w4':
        #         d = self.hl2_node_num
        #     elif w == 'w5':
        #         d = self.hl3_node_num
        #     it = np.nditer(layer_w_t, flags=['f_index'])
        #     while not it.finished:
        #         r = it.index / d
        #         c = it.index % d
        #         w_matrix[i][r][c] = it[0]
        #         it.iternext()
        #     i += 1
        # print w_matrix
        # with open(model_file, 'w') as f:
        #     s = 1
        #     for data_slice in w_matrix:
        #         f.write('#w' + str(s) + '\n')
        #         np.savetxt(f, data_slice, fmt='%2.3f')
        #         s += 1
        # session.close()
        # endregion

    def cross_validate(self, x, y, k, model_file):
        """Split the input_texts data into k parts and do cross-validation
            - Pick 1 part as  test data
            - (k - 1) parts as training data
        """
        # Clear the history
        self.test_accuracy_history = []
        self.train_accuracy_history = []
        self.items_test = 0
        self.items_trained = 0
        row = len(x)
        block = row / k
        # print 'Block:', block
        training_time = timedelta(0, 0, 0)
        testing_time = timedelta(0, 0, 0)
        for i in range(k):
            sl_i = slice(i*block, (i + 1) * block)
            text_x = x[sl_i]
            # print 'Test X:', i, text_x
            test_y = y[sl_i]
            # test_y = np.split(y, [i*k, (i + 1) * k], axis=0)
            # print 'Test Y:', i, test_y
            train_x = np.delete(x, np.s_[i * block: (i + 1) * block], axis=0)
            # print 'Train X:', i, train_x
            train_y = np.delete(y, np.s_[i * block: (i + 1) * block], axis=0)
            # print 'Train Y:', i, train_y
            start_time = datetime.now()
            if i < k - 1:
                print 'Training on ', str(i + 1), 'fold....'
                self.train(train_x, train_y, model_file, False)
            else:
                print 'Training on ', str(i + 1), 'fold....'
                self.train(train_x, train_y, model_file, True)
            duration = datetime.now() - start_time
            training_time += duration
            start_time = datetime.now()
            self.test(text_x, test_y, model_file)
            duration = datetime.now() - start_time
            testing_time += duration
        return training_time, testing_time

    def cross_validate_optimized(self, x, y, k, model_file):
        # Clear the history
        self.test_accuracy_history = []
        self.train_accuracy_history = []
        self.items_test = 0
        self.items_trained = 0
        prediction = tf.nn.softmax(self.nn_output)
        # Backward propagation: update weights to minimize the cost using
        # cross_entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.nn_output, labels=self.y_holder))
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)
        row = len(x)
        block = row / k
        # print 'Block:', block
        training_time = timedelta(0, 0, 0)
        testing_time = timedelta(0, 0, 0)
        items_count = 0
        # Mark the 1000 items step to print number of processed item
        grand = 1
        is_printed = False
        for i in range(k):
            sl_i = slice(i * block, (i + 1) * block)
            text_x = x[sl_i]
            # print 'Test X:', i, text_x
            test_y = y[sl_i]
            # test_y = np.split(y, [i*k, (i + 1) * k], axis=0)
            # print 'Test Y:', i, test_y
            train_x = np.delete(x, np.s_[i * block: (i + 1) * block], axis=0)
            # print 'Train X:', i, train_x
            train_y = np.delete(y, np.s_[i * block: (i + 1) * block], axis=0)
            # print 'Train Y:', i, train_y
            print 'Start training on Si except S[', i, ']'
            start_time = datetime.now()
            for e in range(self.epoch):
                # shuffle the data before training
                for r in range(0, len(train_x)):
                    try:
                        j = random.randint(r + 1, len(train_x) - 1)
                        if r != j:
                            train_x[r], train_x[j] = train_x[j], train_x[r]
                            train_y[r], train_y[j] = train_y[j], train_y[r]
                    except ValueError:
                        pass
                        # print 'End of the list when shuffling'

                # slice the training data into mini batches and train on these batches
                for b in range(0, len(train_x), self.batch_size):
                    batch_x = train_x[b:b + self.batch_size]
                    batch_y = train_y[b:b + self.batch_size]
                    session.run(optimizer,
                                feed_dict={self.input_holder: batch_x,
                                           self.y_holder: batch_y})

                    correct_pred = tf.equal(tf.argmax(prediction, 1),
                                            tf.argmax(self.y_holder, 1))
                    accuracy = tf.reduce_mean(
                        tf.cast(correct_pred, tf.float32))
                    accuracy = session.run(accuracy,
                                           feed_dict={
                                               self.input_holder: batch_x,
                                               self.y_holder: batch_y})

                    self.items_trained += len(train_x)
                    # The homework does not require to calculate the training accuracy
                    # self.train_accuracy_history.append(accuracy)
                    items_count += len(batch_x)
                    jp = items_count / 1000
                    if items_count > (grand * 1000) and not is_printed:
                        print str(items_count - (
                        items_count % 1000)) + ' items processed'
                        is_printed = True
                        grand += 1
                    elif jp >= grand:
                        is_printed = False

            duration = datetime.now() - start_time
            training_time += duration
            print 'Start testing on S[', i, '] data'
            start_time = datetime.now()
            correct_pred = tf.equal(tf.argmax(self.nn_output, 1),
                                    tf.argmax(self.y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy = session.run(accuracy, feed_dict={self.input_holder: text_x,
                                                        self.y_holder: test_y})
            self.items_test += len(text_x)
            self.test_accuracy_history.append(accuracy)

            duration = datetime.now() - start_time
            testing_time += duration
        return training_time, testing_time


    def test_load_manually(self, x, y, model_file):
        """This is the original version which load model file manually"""
        # print 'Before assigning weight:', self.weights['w1']

        full_path = os.path.join(os.path.curdir, model_file)
        w = np.loadtxt(full_path)
        w = np.reshape(w, (5, self.max_node_of_layers, self.number_of_features))
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
            init = tf.global_variables_initializer()

            correct_pred = tf.equal(tf.argmax(self.nn_output, 1), tf.argmax(self.y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            session = tf.Session()
            # with tf.Session as session:
            session.run(init)
            predict_y = session.run(self.nn_output, feed_dict={self.input_holder: x, self.y_holder: y})
            mse = tf.reduce_mean(tf.square(predict_y - y))
            mse = session.run(mse)

            accuracy = session.run(accuracy, feed_dict={self.input_holder: x, self.y_holder: y})
            self.items_test += len(x)
            self.test_accuracy_history.append(accuracy)

            session.close()
        else:
            raise IOError('Weight can not be loaded from model file')

    def test(self, x, y, model_file):
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        # Load checkpoint file which is saved from training step
        saver = tf.train.import_meta_graph(model_file + '.meta')
        saver.restore(session, tf.train.latest_checkpoint('./'))
        # print 'b1: \n', session.run('b1:0')

        # Assign the weights which are loaded from model file
        self.weights['w1'].assign(session.run('w1:0'))
        self.weights['w2'].assign(session.run('w2:0'))
        self.weights['w3'].assign(session.run('w3:0'))
        self.weights['w4'].assign(session.run('w4:0'))
        self.weights['w5'].assign(session.run('w5:0'))
        # Assign the biases which are loaded from model file
        # self.bias['b1'].assign(session.run('b1:0'))
        # self.bias['b2'].assign(session.run('b2:0'))
        # self.bias['b3'].assign(session.run('b3:0'))
        # self.bias['b4'].assign(session.run('b4:0'))
        # self.bias['b5'].assign(session.run('b5:0'))

        correct_pred = tf.equal(tf.argmax(self.nn_output, 1),
                                tf.argmax(self.y_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy = session.run(accuracy, feed_dict={self.input_holder: x,
                                                    self.y_holder: y})
        self.items_test += len(x)
        self.test_accuracy_history.append(accuracy)

        # confusion_matrix_tf = tf.confusion_matrix(tf.argmax(self.nn_output, 1), tf.argmax(self.y_holder, 1))
        # cm = confusion_matrix_tf.eval(feed_dict={self.input_holder: x, self.y_holder: y})
        # confusion = tf.confusion_matrix(labels=self.y_holder, predictions=correct_pred, num_classes=6)
        # print 'Confusion matrix:', session.run(confusion_matrix_tf, feed_dict={self.input_holder: x, self.y_holder: y})
        session.close()

    @property
    def il_node_num(self):
        return self.__il_node_num

    @il_node_num.setter
    def il_node_num(self, value):
        if value <= 0:
            raise ValueError('Input layer: Number of node must greater than 0')
        else:
            self.__il_node_num = value

    @property
    def hl1_node_num(self):
        return self.__hl1_node_num

    @hl1_node_num.setter
    def hl1_node_num(self, value):
        if value <= 0:
            raise ValueError('Hidden layer 1: Number of node must greater than 0')
        else:
            self.__hl1_node_num = value

    @property
    def hl2_node_num(self):
        return self.__hl2_node_num

    @hl2_node_num.setter
    def hl2_node_num(self, value):
        if value <= 0:
            raise ValueError('Hidden layer 2: Number of node must greater than 0')
        else:
            self.__hl2_node_num = value

    @property
    def hl3_node_num(self):
        return self.__hl3_node_num

    @hl3_node_num.setter
    def hl3_node_num(self, value):
        if value <= 0:
            raise ValueError('Hidden layer 3: Number of node must greater than 0')
        else:
            self.__hl3_node_num = value

    @property
    def ol_node_num(self):
        return self.__ol_node_num

    @ol_node_num.setter
    def ol_node_num(self, value):
        if value <= 0:
            raise ValueError('Output layer: Number of node must greater than 0')
        else:
            self.__ol_node_num = value

    @property
    def batch_size(self):
        return self.__mini_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            raise ValueError('Batch size must greater than 0')
        else:
            self.__mini_batch_size = value

    @property
    def max_node_of_layers(self):
        return max(self.il_node_num, self.hl1_node_num, self.hl2_node_num, self.hl3_node_num, self.ol_node_num)

    @property
    def epoch(self):
        return self.__epoch

    @epoch.setter
    def epoch(self, value):
        if value <= 0:
            raise ValueError('Epoch must greater than 0')
        else:
            self.__epoch = value

    @property
    def label_count(self):
        return 6

    @property
    def number_of_features(self):
        return self.__number_of_features

    @property
    def kfold(self):
        return self.__kfold

    @kfold.setter
    def kfold(self, value):
        if 0 < value <= 5:
            self.__kfold = value
        else:
            raise ValueError('Cross validation should be from 1 to 5')

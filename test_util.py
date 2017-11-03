import util
import tensorflow as tf
# from neuron_network import NeuronNetwork

# print util.mutation("A")

# input_data = util.string_to_ascii("BACBDCBDBBACBACDADDCABBCBACDACDDBDABDACD")

# net = NeuronNetwork([[2.0, 1.0]], 1, 1)
# net.il_node_num = 10

# print net.il_node_num, net.hl2_node_num

#test_data = util.load_test_data('data_folder')
#print test_data
# x = tf.placeholder(tf.float32, shape=(3, 3))

x = tf.constant([[1,2],[3,4],[5,6]], tf.float32, shape=(3,2), name='X', verify_shape=(2, 3))
session = tf.Session()
session.run(x)

import sys
import util
from neuron_network import NeuronNetwork

supported_mode = ['train', '5fold', 'test']
BATCH_SIZE = 128

def validate_arguments(arguments):
    if len(arguments) < 4:
        print ('Missing arguments')
        return False
    if not (arguments[1] in supported_mode):
        print 'Invalid mode, supported modes are', supported_mode
        return False
    return True


if __name__ == "__main__":

    train_data = ''
    model_file = ''
    mode = ''
    if validate_arguments(sys.argv):
         train_data = sys.argv[3]
         model_file = sys.argv[2]
         mode = sys.argv[1]
    else:
         sys.exit("Invalid Arguments")
    inputs, outputs = util.load_test_data(train_data)

    nn = NeuronNetwork(inputs, outputs)
    # set the number of node for input layer
    nn.il_node_num = 10
    # set the number of node for hidden layer 1
    nn.hl1_node_num = 10
    # set the number of node for hidden layer 2
    nn.hl2_node_num = 10
    # set the number of node for hidden layer 3
    nn.hl3_node_num = 10
    # set the number of node for out layer
    nn.ol_node_num = 1
    nn.batch_size = BATCH_SIZE
    nn.epoch = 10
    nn.build_network()
    nn.start(mode, model_file)
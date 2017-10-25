import string
import sys
import util
from neuron_network import NeuronNetwork

supported_mode = ['train', '5fold', 'test.txt']


def validate_arguments(arguments):
    if len(arguments) < 4:
        print ('Missing arguments')
        return False
    if not (arguments[1] in supported_mode):
        print 'Invalid mode, supported modes are', supported_mode
        return False
    # All the test.txt passed
    return True


if __name__ == "__main__":
    # Test
    # input_str = "BACBDCBDBBACBACDADDCABBCBACDACDDBDABDACD"
    # result = determine_sticky(input_str)
    # print (result)
    # args = sys.argv
    # mode = args[1]
    # model_file.txt = args[2]
    # data_file = args[3]
    # if validate_arguments(args):
    #     print 'Arguments are valid,', mode, model_file.txt, data_file
    # inputs[0] : NON-STICK
    # inputs[1] : 12-STICK
    # inputs[2] : 34-STICK
    # inputs[3] : 56-STICK
    # inputs[4] : 78-STICK
    # inputs[5] : PALINDROME - STICK

    inputs = ['BACBDCBDBBACBACDADDCABBCBACDACDDBDABDACD',
              'BACBDCBDBBACBACDADDCABBCBACDACDDBDABDAAD',
              'BACDDCBDBBACBACDADDCABBCBACDACDDBDABDACD']
    features = []

    for i in inputs:
        e = util.string_to_ascii(i)
        features.append(e)
    outputs = [[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]

    nn = NeuronNetwork(features, outputs, 0.01)
    nn.batch_size = 1
    # nn.start('train', 'model_file.txt', 'data_folder')
    nn.start('test.txt', 'model_file.txt', 'data_folder')

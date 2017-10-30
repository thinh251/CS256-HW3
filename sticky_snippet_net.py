import string
import sys
import numpy as np
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
    # input_texts[0] : NON-STICK
    # input_texts[1] : 12-STICK
    # input_texts[2] : 34-STICK
    # input_texts[3] : 56-STICK
    # input_texts[4] : 78-STICK
    # input_texts[5] : PALINDROME - STICK

    # input_texts = ['BACBDCBDBBACBACDADDCABBCBACDACDDBDABDACD',
    #                'BACBDCBDBBACBACDADDCABBCBACDACDDBDABDAAD',
    #                'BACDDCBDBBACBACDADDCABBCBACDACDDBDABDACD']

    # for i in input_texts:
    #     e = util.string_to_ascii(i)
    #     inputs.append(e)
    # outputs = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    # test_size = 10000
    # test_number_of_features = 40
    # test_number_of_labels = 6
    # inputs = np.random.randint(low=65, high=68, size=(test_size, test_number_of_features))
    # outputs = np.random.randint(low=0, high=1, size=(test_size, test_number_of_labels))
    # print 'outputs:', outputs
    # print 'Output:', outputs

    inputs, outputs = util.load_test_data('data_folder')
    nn = NeuronNetwork(inputs, outputs, 0.01)
    nn.batch_size = 128
    nn.epoch = 5
    nn.start('train', 'model_thinh.txt', 'data_folder')
    # nn.start('test', 'model_thinh.txt', 'data_folder')
    # nn.start('5fold', 'model_file', 'data_folder')

import string
import sys
import util
from neuron_network import NeuronNetwork

supported_mode = ['train', '5fold', 'test']


def validate_arguments(arguments):
    if len(arguments) < 4:
        print ('Missing arguments')
        return False
    if not (arguments[1] in supported_mode):
        print 'Invalid mode, supported modes are', supported_mode
        return False
    # All the test passed
    return True


def train():
    # TODO
    print 'Train function'


def five_fold():
    # TODO
    print '5fold function'


def test():
    # TODO:
    print 'Test function'


def determine_sticky(input_str):
    """Determine the sticky label of a string"""
    if input_str is None or len(input_str) == 0:
        return

    # v, w = input_str[:len(input_str)/2], input_str[len(input_str)/2:]
    # w_reverse = w[::-1]  # reverse w
    k = 1
    is_stick = True
    while is_stick and k <= len(input_str)/2:
        u = input_str[:k]
        v_len = len(input_str) - 2*k
        w = input_str[v_len + k:]
        w_reverse = w[::-1]
        is_stick = util.sticks_with(u, w_reverse)
        if is_stick:
            k += 1

    if k == 1:
        return 'NONSTICK'
    elif k < len(input_str)/2:
        return str(k - 1) + str(k) + '-STICKY'
    else:
        return 'STICK_PALINDROME'


if __name__ == "__main__":
    # Test
    #input_str = "BACBDCBDBBACBACDADDCABBCBACDACDDBDABDACD"
    #result = determine_sticky(input_str)
    # print (result)
    # args = sys.argv
    # mode = args[1]
    # model_file = args[2]
    # data_file = args[3]
    # if validate_arguments(args):
    #     print 'Arguments are valid,', mode, model_file, data_file
    # inputs[0] : NON-STICK
    # inputs[1] : 12-STICK
    # inputs[2] : 34-STICK
    # inputs[3] : 56-STICK
    # inputs[4] : 78-STICK
    # inputs[5] : PALINDROME - STICK

    inputs = ['BACBDCBDBBACBACDADDCABBCBACDACDDBDABDACD', 'BACBDCBDBBACBACDADDCABBCBACDACDDBDABDAAD',
              'BACDDCBDBBACBACDADDCABBCBACDACDDBDABDACD']
    features = []

    for i in inputs:
        e = util.string_to_ascii(i)
        features.append(e)
    outputs = [[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]

    nn = NeuronNetwork(features, outputs, 0.01)
    nn.batch_size = 1
    nn.start('train', 'model_file')

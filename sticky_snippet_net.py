import string
import sys
import util

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
    input_str = "BACBDCBDBBACBACDADDCABBCBACDACDDBDABDACD"
    result = determine_sticky(input_str)
    print (result)
    # args = sys.argv
    # mode = args[1]
    # model_file = args[2]
    # data_file = args[3]
    # if validate_arguments(args):
    #     print 'Arguments are valid,', mode, model_file, data_file

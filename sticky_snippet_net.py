import sys

supported_mode = ['train', '5fold', 'test']


def validate_arguments(arguments):
    if len(arguments) < 4:
        print ('Missing arguments')
        return False
    if not (arguments[1] in supported_mode):
        print 'Mode is incorrect, supported modes are', supported_mode
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


if __name__ == "__main__":
    args = sys.argv
    mode = args[1]
    model_file = args[2]
    data_file = args[3]
    if validate_arguments(args):
        print 'Arguments are valid,', mode, model_file, data_file

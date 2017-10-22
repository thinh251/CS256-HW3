import numpy as np
import os


def sticks_with(u,v):
    """returns 1 if two strings stick. 0 otherwise."""
    if len(u) != len(v): # length should be same for strings to stick.
        return 0
    else:
        U = list(u)
        V = list(v)
        for i in range(len(v)):
            if U[i] == 'A':
                if V[i] != 'C':
                    return 0
            if U[i] == 'B':
                if V[i] != 'D':
                    return 0
            if U[i] == 'C':
                if V[i] != 'A':
                    return 0
            if U[i] == 'D':
                if V[i] != 'B':
                    return 0
        return 1


def sticky_type(input):
    for k in range(len(input)/2):
        if sticks_with(input[k],input[-k-1]) == 0:
            return sticky_index(k)


def sticky_index(k):
    if k == 0 : return "Non-Stikcy"
    if k in [1,2]: return "12-STICKY"
    if k in [3,4]: return "34-STICKY"
    if k in [5,6]: return "56-STICKY"
    if k in [7,8]: return "78-STICKY"
    else: return "STICK_PALINDROME"


def mutation(letter):
    """returns a mutation for a letter randomly among the other letters"""
    letters = "ABCD"
    random_index = np.random.randint(3)
    return letters.replace(letter,"")[random_index]


def write_weights_file(model_file, data):
    f = open(model_file, 'w')
    f.write(data)
    f.close()
    print 'Data written to file:', model_file


def read_weights_from(model_file):
    f = open(model_file, 'r')
    weights = f.read()
    return weights


def check_line_length(text_file, n):
    """Check length of individual line in file if a length is != n,
        skip the file
    """
    with open(text_file) as f:
        for line in f:
            if len(line.strip()) != n:
                return False
        return True


def read_data(data_folder):
    files = os.listdir(data_folder)
    something = []
    for f in files:
        if os.path.isfile(f) and check_line_length(f, 40):
            # check the text and label it
            with open(f) as text_file:
                for line in text_file:
                    t = sticky_type(line)
                    # TODO:
                    something.append([line, t])
    return something








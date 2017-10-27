import glob
import os

import numpy as np


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
    files = glob.glob(data_folder)
    something = []
    lines = []
    for f in files:
        if os.path.isfile(f) and check_line_length(f, 40):
            # check the text and label it
            with open(f) as text_file:
                for line in text_file:
                    lines.append(line)
                    t = determine_sticky(line)
                    lines.append(t)
                    # TODO:
                    something.append(lines)
    return something


def string_to_ascii(text):
    if text is None or len(text) == 0:
        return None
    features = []
    for s in text:
        features.append(ord(s))
    return features


def numberize_output_label(text):
    if text == 'NONSTICK':
        return [1, 0 ,0, 0, 0, 0]
    elif text == "12-STICKY":
        return [0, 1, 0, 0, 0, 0]
    elif text == "34-STICKY":
        return [0, 0, 1, 0, 0, 0]
    elif text == "56-STICKY":
        return [0, 0, 0, 1, 0, 0]
    elif text == "78-STICKY":
        return [0, 0, 0, 0, 1, 0]
    elif text == "STICK_PALINDROME":
        return [0, 0, 0, 0, 0, 1]

#loads data from the file
def load_test_data(data_folder):
    path = os.path.curdir + '/' + data_folder + '/*.txt'
    test_data = read_data(path)
    test_x = []
    test_y = []
    for t in test_data:
        text = t[0]
        label = t[1]
        features = string_to_ascii(text)
        test_x.append(features)
        output = numberize_output_label(label)
        test_y.append(output)
    return test_x, test_y

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
        is_stick = sticks_with(u, w_reverse)
        if is_stick:
            k += 1

    if k == 1:
        return 'NONSTICK'
    elif k < len(input_str)/2:
        return str(k - 1) + str(k) + '-STICKY'
    else:
        return 'STICK_PALINDROME'

def get_correct_match(u):

    """Returns match for the char"""
    if u == 'A':
        return 'C'
    if u == 'B':
        return 'D'
    if u == 'C':
        return 'A'
    if u == 'D':
        return 'B'

def get_incorrect_match(u):
    """Returns incorrect match for the char"""
    random_num = np.random.randint(3)
    if u == 'A':
        if random_num == 0 :
            return 'A'
        if random_num == 1:
            return 'B'
        if random_num == 2:
            return 'D'
    if u == 'B':
        if random_num == 0 :
            return 'A'
        if random_num == 1:
            return 'B'
        if random_num == 2:
            return 'C'
    if u == 'C':
        if random_num == 0 :
            return 'B'
        if random_num == 1:
            return 'C'
        if random_num == 2:
            return 'D'
    if u == 'D':
        if random_num == 0 :
            return 'A'
        if random_num == 1:
            return 'C'
        if random_num == 2:
            return 'D'

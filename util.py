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
        return 1;


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
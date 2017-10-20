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

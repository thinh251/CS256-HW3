def sticks_with(u,v):
    if len(u) != len(v):
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



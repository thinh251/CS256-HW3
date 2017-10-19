import numpy as np
import sys
import util

if len(sys.argv) < 5:
    sys.exit("gene_snippet_generator takes 5 arguments.")
try:
    num_snippets = int(sys.argv[1])
    mutation_rate = float(sys.argv[2])
    from_ends = int(sys.argv[3])
    output_file =  sys.argv[4]
except:
    sys.exit("invalid argument !")

letters = ['A','B','C','D']

for i in range(num_snippets+1):
    first_rand=np.random.randint(0, 4, 10)  #generate first half randomly
    gene_code = []
    for ele in first_rand:
        gene_code.append(letters[ele])
    for j in range(from_ends):
        random_float = np.random.uniform()
        if random_float <= mutation_rate :
            gene_code.append(util.get_incorrect_match(gene_code[j]))
        else :
            gene_code.append(util.get_correct_match(gene_code[j]))
    for j in range(from_ends,10):
        random_float = np.random.uniform()
        if random_float <= 0.5 :
            gene_code.append(util.get_incorrect_match(gene_code[j]))
        else :
            gene_code.append(util.get_correct_match(gene_code[j]))

    print ''.join(gene_code)
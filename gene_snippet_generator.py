import numpy as np
import sys
import util

gene_len = 40

if len(sys.argv) < 5:
    sys.exit("gene_snippet_generator takes 5 arguments.")
try:
    num_snippets = int(sys.argv[1])
    mutation_rate = float(sys.argv[2])
    from_ends = int(sys.argv[3])
    output_file =  sys.argv[4]
except:
    sys.exit("invalid argument !")

output = ''
letters = ['A','B','C','D']

for i in range(num_snippets):
    first_rand=np.random.randint(0, 4, gene_len/2)  #generate first half
    # randomly
    gene_code = []
    for ele in first_rand: # Generates letters for first 20 chars
        gene_code.append(letters[ele])
    for j in range(gene_len/2-1,from_ends,-1):#generates letters till
        # 'from_ends' with mutation probablity as 1/2
        random_float = np.random.uniform()
        if random_float <= 0.5 :
            gene_code.append(util.get_incorrect_match(gene_code[j]))
        else :
            gene_code.append(util.get_correct_match(gene_code[j]))
    for j in range(from_ends,0,-1): # generates letters from from ends till
        # last word according to the mutation rate given
        random_float = np.random.uniform()
        if random_float <= mutation_rate :
            gene_code.append(util.get_incorrect_match(gene_code[j]))
        else :
            gene_code.append(util.get_correct_match(gene_code[j]))


    output = output + ''.join(gene_code)+'\n'

print output
F = open(output_file,'w')
F.write(''.join(output))
F.close()
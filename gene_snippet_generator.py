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
    #create a second half perfect palinfrom
    #then do the mutations
    gene_code = []
    for element in first_rand: # Generates letters for first 20 chars
        gene_code.append(letters[ele])
    #generating the second half to make the string palindrome
    gene_code_str = ''.join(gene_code) + ''.join(reversed(gene_code))
    for j in range(from_ends,len(gene_code_str)-from_ends):#generates letters till
        # does the 50% mutation on the further than from_end distance
        random_float = np.random.uniform()
        if random_float <= 0.5 :
            gene_code_str.replace(j,util.mutation(gene_code_str[j]))
    for j in range(-from_ends,from_ends): # generates letters from from ends till
        # first and last sections when within the distance from start and end
        random_float = np.random.uniform()
        if random_float <= mutation_rate :
            gene_code_str.replace(j,util.mutation(gene_code_str[j]))


    output = output + gene_code_str+'\n'

print output
F = open(output_file,'w')
F.write(''.join(output))
F.close()

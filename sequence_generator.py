# A generator for sequence based on which terms are permitted in the sequence

from function import Function, FunctionTerm
import random
import numpy as np

def make_possible_terms():
    """Generates a collection of all the possible individual terms. When generating functions, we randomly select terms from this list and assign random integer coefficients to it.
    """

    possible_terms = []

    # 1, n, n^2 and n^3
    possible_terms.extend([
        FunctionTerm(type='constant'),
        FunctionTerm(type='loc_term', exponent1=1),
        FunctionTerm(type='loc_term', exponent1=2),
        FunctionTerm(type='loc_term', exponent1=3),
    ])

    # Power terms
    for index_diff in range(1, 4):
        for power in range(1, 3):
            possible_terms.append(FunctionTerm(type='power_term', exponent1=power, index_diff1=index_diff))

    # Interaction terms - this one generates a lot of terms, might consider turning
    # it off first
    # for index_diff1 in range(1, 3):
    #     for index_diff2 in range(index_diff1, 4): # multiplication is commutative
    #         for exponent1 in range(1, 3):
    #             for exponent2 in range(1, 3):
    #                 possible_terms.append(FunctionTerm(type='interaction_term',
    #                 exponent1=exponent1, exponent2=exponent2,
    #                 index_diff1=index_diff1, index_diff2=index_diff2))

    return possible_terms

def make_random_function(sequence_bound=10000, nterms=5, coefficient_range=(1, 5)):
    # Find a function with max(sequence) <= sequence_bound
    # Returns (function, sequence, terms_used)
    # terms_used is a boolean array

    possible_terms = make_possible_terms() # Make a copy of this list every iteration

    while True: # this process sometimes fails
        f = Function()
        term_choices = random.sample(range(len(possible_terms)), k=nterms)
        for term_index in term_choices:
            t = possible_terms[term_index]
            t.updateCoeff(random.randint(*coefficient_range)) # randomly assign coefficient
            f.addTerm(t)

        sequence = make_sequence(f)
        if max(sequence) > sequence_bound: continue
        else:
            boolean_term_mask = np.zeros(len(possible_terms), dtype=bool)
            boolean_term_mask[term_choices] = True
            return f, sequence, boolean_term_mask


### DEPRECATED: USE make_random_function() INSTEAD
def make_functions(
    num_functions_generated=10,
    num_terms_mean=4, num_terms_stdev=2, min_num_terms=2,
    coefficient_range=(1, 5)
    ):
    """
    DEPRECATED: USE make_random_function() INSTEAD

    Returns a list of (functions, terms used) found by concatenating terms given by make_possible_terms(). The number of terms is drawn from a Gaussian distribution.

    terms_used is expressed as a list of indices to the output of make_possible_terms().

    Args:
        num_functions_generated (int, optional): Number of functions to generate. Defaults to 10.
        num_terms_mean (int, optional): Average number of terms. Defaults to 4.
        num_terms_stdev (int, optional): Standard deviation of number of terms. Defaults to 2.
        min_num_terms (int, optional): Each function returned with have at least this many terms. Defaults to 2.
        coefficient_range (tuple, optional): Range of coefficients before each term. Defaults to (1, 5).
    """

    # print(f'Params passed to make_functions(): {locals()}')
    # possible_terms = make_possible_terms()
    # print(f'Possible terms: {possible_terms}')

    out = [] # the list of functions to return
    while len(out) != num_functions_generated:
        possible_terms = make_possible_terms() # Make a copy of this list every iteration

        # Select n terms uniquely
        nterms = round(random.gauss(num_terms_mean, num_terms_stdev))

        if nterms < min_num_terms:
            continue
        if nterms > len(possible_terms):
            continue


        f = Function()
        term_choices = random.sample(range(len(possible_terms)), k=nterms)
        for term_index in term_choices:
            t = possible_terms[term_index]
            t.updateCoeff(random.randint(*coefficient_range)) # randomly assign coefficient
            f.addTerm(t)

        out.append((f, term_choices))

    return out

def make_sequence(
    function: Function,
    num_generated_terms = 10,
    initial_terms_range = (1, 3)
    ):
    """Makes a sequence using the given function. Randomly generates the first few terms where the function is invalid, the index of which is given by Function.startIndex().

    Args:
        function (Function): Function used
        num_generated_terms (int, optional): Number of terms to generate. Defaults to 10.
        initial_terms_range (tuple, optional): Range of the initial terms. Defaults to (1, 3).

    Returns:
        list[int]: _description_
    """

    sequence = []
    # We first hallucinate the first few terms
    for i in range(function.startIndex()-1):
        sequence.append(random.randint(*initial_terms_range))

    while len(sequence) != num_generated_terms:
        # evaluate the function on previous terms
        sequence.append(function.evaluate(sequence, len(sequence) + 1))

    return sequence



def run():
    # print(f'Functions generated: {make_functions()}')

    seqs = []
    f = Function()
    f.addTerm(FunctionTerm(type='loc_term', c=1, exponent1=2))
    print(make_sequence(f))

    fs = make_functions(1000)

    for f, _ in fs:
        generated_sequence = make_sequence(f)
        if max(make_sequence(f)) < 200:
            print(f)
            print(generated_sequence)
            print()
            seqs.append(generated_sequence)
    
    return seqs
  
if __name__ == '__main__':
    # f = Function()
    # f.addTerm(FunctionTerm(type='constant', c=2))
    # f.addTerm(FunctionTerm(type='loc_term', c=1, exponent1=1, index_diff1=1))
    # print(f)
    # print(make_sequence(f))
    # # run()
    print(make_random_function())
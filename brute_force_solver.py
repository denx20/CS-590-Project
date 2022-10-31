from copy import deepcopy
from function import Function, FunctionTerm


# TODO: either manually make this list longer, or convert it into a generator
candidate_term_list = [
  FunctionTerm('constant'),
  FunctionTerm('loc_term', exponent1=1), # n
  FunctionTerm('power_term', index_diff1=1), # f[n-1]
  FunctionTerm('power_term', index_diff1=2), # f[n-2]
  FunctionTerm('loc_term', exponent1=2), # n^2
  #FunctionTerm('power_term', index_diff1=3), # f[n-3]
  FunctionTerm('loc_term', exponent1=3), # n^3
  FunctionTerm('interaction_term', exponent1=1, exponent2=1, index_diff1=1, index_diff2=2) # f[n-1]*f[n-2]
]



def brute_force_grid_search(sequence, upper_bound = 5, lower_bound = -5, break_threshold=7):
  # constant term, polynomial of index, previous term, previous two terms, ...
  coeff = sorted(list(range(upper_bound, lower_bound-1, -1)), key=lambda x: abs(x))
  base = len(coeff)
  digit_to_coeff = {i: coeff[i] for i in range(base)}
  
  i = 0

  def int_to_base_helper(num, base):
    ret = []
    while num >= base:
      ret.append(num % base)
      num = num // base
    return ret
  
  while True:
    for j in range(base**i, base**(i+1)):
      if j == 0:
        continue
      term_coeffs = [digit_to_coeff[c] for c in int_to_base_helper(j, base)]+[0]*50
      f = Function()
      for k, term in enumerate(candidate_term_list):
        if term_coeffs[k] != 0:
          term_copy = deepcopy(term)
          term_copy.updateCoeff(term_coeffs[k])
          f.addTerm(term_copy)
      
      if f.startIndex() > len(sequence):
        continue
      perfect_fit = True
      for n in range(f.startIndex(), len(sequence)+1):
        if f.evaluate(sequence, n) != sequence[n-1]:
          perfect_fit = False
          break
      if perfect_fit:
        return f
    i += 1

    print('i =',i)

    if i >= break_threshold:
      print(f'Search terminated! Searched {base}^{break_threshold} combinations but cannot find solution')
      return None






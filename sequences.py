# In the future, we can expand this script to pull from a txt file
# https://stackoverflow.com/questions/5991756/programmatic-access-to-on-line-encyclopedia-of-integer-sequences



# This is a list of 0-indexed sequences (the 1st entry is the 1st in the array representing each sequence, thus with index 0)
# These problems should be very easy and do not require more than 5 training terms

prototype_sequences = [
  # Sequences that contain at least one formula using only addition, multiplication, exponentiation to at most 3, modulo 2, and 2-morphisms of previous terms
  
  [	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], #

  [	0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431], # A000217. The solution that does not require division is a(n) = a(n-2) + 2*n - 1

  [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841], # A000290

  [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000], # cubes

  [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89], # fib

  [2, 3, 6, 18, 108, 1944, 209952], # multiplication by prev two terms

  [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], # simple use of the modulus

  [2, 1, 4, 3, 6, 9, 8, 27, 10] # from Tony
  
]


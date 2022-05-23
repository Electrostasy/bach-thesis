from itertools import permutations
from string import printable

for c in permutations(printable[0:-5], 5):
  print(''.join(c))

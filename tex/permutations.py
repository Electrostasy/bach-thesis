from itertools import permutations
from string import printable

for i in range(1, 7):
  for c in permutations(printable[0:-5], i):
    print(''.join(c))

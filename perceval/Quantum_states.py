import perceval as pcvl
# or you can import each symbol, depending on your prefered coding style
from perceval import BasicState, StateVector, SVDistribution, BSDistribution, BSCount, BSSamples

## Syntax of different BasicState (list, string, etc)
bs1 = BasicState([0, 2, 0, 1])
bs2 = BasicState('|0,2,0,1>')  # Must start with | and end with >

print(bs1)
print(f"Number of modes: {bs1.m}")
print(f"Number of photons: {bs1.n}")

if bs1 == bs2:
    print("bs1 and bs2 are the same states")

## You can iterate on modes
for i, photon_count in enumerate(bs1):
    print(f"There is {photon_count} photon in mode {i} (or equivalently bs1[{i}]={bs1[i]}).")
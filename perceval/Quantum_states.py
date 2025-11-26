import perceval as pcvl
# or you can import each symbol, depending on your prefered coding style
from perceval import BasicState, StateVector, SVDistribution, BSDistribution, BSCount, BSSamples


##################################
#
#### BASIC STATE
#
##################################

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


# There are three kind of BasicStates

photon_position = [0, 2, 0, 1]

# FockState, all photons are indistinguishable
bs1 = BasicState(photon_position)  # Or equivalently BasicState("|0,2,0,1>")
print(type(bs1), isinstance(bs1, BasicState))

# NoisyFockState, photons with the same tag are indistinguishable, photons with different tags are distinguishable (they will not interact)
noise_index = [0, 1, 0]
bs2 = BasicState(photon_position, noise_index)  # Or equivalently BasicState("|0,{0}{1},0,{0}>")
print(type(bs2), isinstance(bs2, BasicState))

# AnnotatedFockState, with custom annotations (not simulable in the general case, needs a conversion to something simulable first)
bs3 = BasicState("|0,{lambda:925}{lambda:925.01},0,{lambda:925.02}>")
print(type(bs3), isinstance(bs3, BasicState))



# Basic methods are common between these types
print("Reminder, bs1 =", bs1)
print("bs1 * |1,2> =", bs1 * BasicState([1, 2]))  # Tensor product
print("A slice of bs1 (extract modes #1 & 2) =", bs1[1:3])  # Slice
print("bs1 with threshold detection applied =", bs1.threshold_detection())  # Apply a threshold detection to the state, limiting detected photons to 1 per mode


##################################
#
#### STATE VECTOR (superposition)
#
##################################


# StateVectors can be defined using arithmetic on BasicStates and other StateVectors
sv = (0.5 + 0.3j) * BasicState([1, 0, 1, 1]) + BasicState([0, 2, 0, 1])

# State vectors normalize themselves upon use
print("State vector is normalized upon use and display: ", sv)

for (basic_state, amplitude) in sv:
    print(basic_state, "has the complex amplitude", amplitude)

# We can also access amplitudes as in a dictionary
print(sv[pcvl.BasicState([0, 2, 0, 1])])


##################################
#
#### SVDistribution (mixes state)
#
##################################


# A SVDistribution is a collection of StateVectors
svd = SVDistribution({StateVector([1, 2]) : 0.4,
                      BasicState([3, 0]) + BasicState([2, 1]) : 0.6})

print("Five random samples according to the distribution:", svd.sample(5))

svd2 = SVDistribution({StateVector([0]) : 0.1,
                      BasicState([1]) + BasicState([2]) : 0.2})
svd2.normalize()  # distributions have to be normalized to make sense


print(svd)
print(svd2)

print("Tensor product")
print(svd * svd2)  # Tensor product between distributions



##################################
#
#### Results
#
##################################


# The BSDistribution is a collection of FockStates
bsd = BSDistribution()
bsd[BasicState([1, 2])] = 0.4
bsd[BasicState([2, 1])] = 0.2

print(bsd)
print("Number of modes:", bsd.m)

bsd.normalize()  # Not automatic on distributions

# Distributions act much like python dictionaries
for (state, probability) in bsd.items():
    print(state, "has the probability", probability, f"(or equivalently {bsd[state]})")

print("Tensor product")
print(bsd * BasicState([1])) # Tensor product, also works between distributions

bsc = BSCount()
bsc[BasicState([1, 2])] = 20
bsc[BasicState([2, 1])] = 118
print(bsc)
print("Total number of samples:", bsc.total())
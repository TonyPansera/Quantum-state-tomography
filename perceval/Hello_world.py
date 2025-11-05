import perceval as pcvl
from perceval.algorithm import Sampler

input_state = pcvl.BasicState("|1,1>")  # Inject one photon on each input mode...
circuit = pcvl.BS()                     # ... of a perfect beam splitter
noise_model = pcvl.NoiseModel(transmittance=0.95, indistinguishability=0.85)  # Define some noise level

processor = pcvl.Processor("SLOS", circuit, noise=noise_model)  # Use SLOS, a strong simulation back-end
processor.min_detected_photons_filter(1)  # Accept all output states containing at least 1 photon
processor.with_input(input_state)

sampler = Sampler(processor)
samples = sampler.sample_count(10_000)['results']  # Ask to generate 10k samples, and get back only the raw results
probs = sampler.probs()['results']  # Ask for the exact probabilities

print(f"Samples: {samples}")
print(f"Probabilities: {probs}")
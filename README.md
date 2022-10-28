# OptimalTransportDemos
A few short demos illustrating some properties of the Wasserstein-2 distance. 

# Usage
```python wass.py desired_resolution```
where **desired_resolution** tells you how many transformations occur.

# Convexity with Shift
This test is done through ```do_shift_test()```

#Convexity with Partial Amplitude Change
This test is done through ```do_amp_test()```. NOTE: This property is also held by the L^2 norm, and the current status of the code shows that W2 is 
quite sensitive to this. This could be caused by my choice of renormalization operator. If one modifies this to a Gaussian mixture, I believe it would
work, since then renormalization is not necessary or simply is dividing by a number, rather than splitting into positive and negative parts
(a non-smooth operation).

#Robustness with Respect to Noise
This test is done through ```do_noise_test()```.

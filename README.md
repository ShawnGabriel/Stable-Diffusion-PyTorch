# Stable Diffusion from Scratch in PyTorch

We model our data as a big joint distribution. Then, we learn the parameters through neural networks. Goal: learn the complex distribution so we can generate new data.

### The Math
Forward Process 'q': Markov chain of noisification which is a series of Gaussians that add noise. Overall, the forward process involves parameters used in our mean and variance are already known to us, as we initially start with our original image which we add noise little by little until it becomes fully noisified.

Reverse Process 'p': On the other hand, the parameters used for the mean and variance are unknown. The mean is calculated by neural networks and the variance is parameterized by. So, we start off with something noisy, then we would want the next image to be less noisy

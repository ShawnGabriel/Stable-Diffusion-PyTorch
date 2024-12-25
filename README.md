# Stable Diffusion from Scratch in PyTorch

We model our data as a big joint distribution. Then, we learn the parameters through neural networks. Goal: learn the complex distribution so we can generate new data.

### The Math
Forward Process 'q': Markov chain of noisification which is a series of Gaussians that add noise. Overall, the forward process involves parameters used in our mean and variance are already known to us, as we initially start with our original image which we add noise little by little until it becomes fully noisified.

Reverse Process 'p': On the other hand, the parameters used for the mean and variance are unknown. The mean is calculated by neural networks and the variance is parameterized by. So, we start off with something noisy, then we would want the next image to be less noisy.

A lower bound for the quantity Theta is called the Evidence Lower Bound (ELBO). If we maximize the lower bound, it will maximize the likelihood of our data. We do this trhough training a network called ϵ_θ that given a noisy image at time stamp t and the time stamp when the noise was added (also t), it could predict how much noise is in the noisified image. If we do gradient descent on the loss function, we will maximize the ELBO and the likelihood of our data.

How can we create images that match our specific requirements?
Starting from the pure noise, we introduce a signal called a prompt/context which influences the model on how to remove the noise so the output move towards what we want. There's a reason as to why we don't want a prompt during the training of our model. It makes the model a conditioned model, where it's only capable of denoisifying images related to our prompt. On the other hand, an unconditioned model is where there is no prompt at all. The use of both a conditioned and an unconditioned model is called **classifier guidance**.

How to condition the reverse process?
If we instead replace the prompt with a zero, we would make the model both a conditioned and an unconditioned model, telling it to denoisfy whatever is present on the image on its own; this is called **classifier-free guidance**.
output = w * (output_conditioned - output_unconditioned) + output_unconditioned
where w is a weight that indicates how much we want the model to pay attention to the conditioning signal (prompt). The higher this value, the more our output will resemble our prompt.

CLIP (Contrastive Language-Image Pre-training)
A model built by OpenAI that allowed to connect text with images. So in the CLIP, we only utilize text embeddings for Stable Diffusion, as a conditioning signal for our model to denoise the image into what we want.

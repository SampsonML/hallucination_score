# Hallucination score routine for "Spotting Hallucinations in Inverse Problems with Data-Driven Priors"
[![Link - Paper](https://img.shields.io/badge/Link-Paper-blue)](https://arxiv.org/abs/2306.13272)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Note that a full implementation of this routine is used in the mutiband deblending tool Scarlet2 (https://github.com/pmelchior/scarlet2).

We present a method for efficiently calculating an approximation the the diagonal Hessian matrix. This allows for the quantification of uncertainties (hallucinations) in inverse problems in the domain of image generation. An example routine using astronomy data is shown below,
<img src="/images/fig1.png" height="400">

First we perform a source seperation, and regenerate a galaxy image with a combination of a data source and a data-driven prior in the form of a score-based neural network. With two disprate signals for the data (likelhihood) and the neural network (prior) we are able to quantify the uncertainy in the reconstruction via taking a Hessian matrix for each. For computational efficiency we use the Hutchinson method and the Hessian-vector product routine in JAX to compute an approximate diagonal Hessian with only O(n) computations. 

We can see the values for the approximated diagonal Hessians in columns 3 and 4. The "hallucination score" is simple the negation of the diagonal Hessian of the likelihood from the diagonal Hessian of the prior.
<img src="/images/fig2.png" height="400">

// backticks are not allowed..
let textdata = String.raw`

# Fast Diffusion for Quasigeostrophic Flows
## diffusion models at the speed of psuedospectral solvers

Akhil Sadam, 2026-5-12 [MIT 6.8300 CV Project]




<!-- 

## Motivation

Deep learning (DL) for advective PDEs is widely researched, but current models are both enormous and somewhat inefficient.
 
- Current models fail to capture hidden or non-observable states.
- Current models ignore the multiscale / fractal nature of the problem. 

Here we define multiscale as the presence of coupled PDEs acting at different spatial scale, and fractal as the presence of self-similarity among scales (in-particular fixed-point solutions) in the solution manifold.

### Non-observable states
Current approaches for the former include:

- Forecasting with delay-states as in cite{SINDy}, 
cite{SpacetimePOD}, cite{Attention}
- Forecasting with memory as in cite{FuXi}

In all cases, a kernel integration/convolution over time is approximated, either implicitly with past trajectories or explicitly with a large neural network. On large weather data, though, computation is intractable: models like cite{FuXi} only use a memory of 2 snapshots. 

To adequately provide context, the single-state model must be small enough to accommodate memory terms. To this end, we investigate methods to compress the multiple scales.
For problems with multiple, coupled scales, current PDE-specific ML is focused on basis decomposition. Neural ODE models such as cite{ClimODE}, cite{NeuralGCM} do not separate scales, and are either not efficient or competitive. 
Arguably, the SOTA small model approach for weather includes Fourier Neural Operator variations (cite{FNO}, cite{AFNO}, cite{SFNO}) which rely on a Fourier basis decomposition---still too large to include sufficient memory*.

### Multiscale nature
We address this memory issue by compressing the PDE solution manifold.
While commonly done as a 'superresolution' technique, we focus on scale-invariance, so as to combine mutiple scales.
Proper scale-invariance can produce a network that can compress and decompress at arbitrary resolution, and deblur, denoise, and superresolve data.
As such, network design is much more difficult.

Current autoencoder approaches include Convolutional U-Net cite{UNet}, Vision Transformer cite{ViT}, and cite{Diffusion} architectures, but these compress scales independently, or preserve only the overall distribution. 

We propose a simple first approach to a deterministic, scale-invariant autoencoder, and investigate emergent behavior.

In particular, we
- Enforce scale-invariance in network structure
- Investigate constrained latent-space effects (as in variational autoencoders or cite{VAE}) on training and performance 
- Identify pathological features with highly nonlinear scale dependence
- Identify scale-invariant domain-specific features (a latent basis that breaks the Kolmogorov barrier)
- Show a maximum feasibility limit by considering limiting-sequences in the intrinsic-dimension limit.

## Minimum Entropy Regularization
By generic information theory, the universal compression problem is equivalent to any prediction problem (compression of joint probability distributions given a parametric family of priors, or compress the conditionals given the prior family again).

Further, cross-entropy loss is equivalent to reverse-KL divergence, and it can be shown that the minimum entropy / most compressed data solution is the minimizer of the reverse-KL loss (see Polyanskiy & Wu's cite{InformationTheory} text).

The distance-to-maximum-compression also predictes the generalization gap, as found in this paper cite{PACBayesGeneralize}.

So we add a minimum entropy constraint to the latent space, to enforce the most compressed solution. This is known as the minimum description length principle, MDL; Occam's Razor represents the same idea.

The difficulty: we do not have a probability measure on the latent space.
For now, since the data is wave-like in nature, we pretend it is a wavefunction.

Formally, we minimize the entropy of the energy ($|v|^2$) in the latent space.
This is reasonably well motivated from statistical mechanics (see Equipartition Principle, which is the opposite), but is still relatively arbitrary; for example, the variance could be minimized instead.***

Further research will investigate other probability surrogates.

## Intrinsic Dimension [Two-NN Method]
First, recall the definition of intrinsic dimension: we seek to fit the local tangent plane to the solution manifold, and count the dimensions of this tangent plane.

We follow the Two-NN method, where a simple volume ratio calculation using distances to the 1st and 2nd nearest neighbors estimates the intrinsic dimension.
Please see cite{TwoNN} for more details.

## Architecture

### Data
First, a quick mention of the data: we use two advective test cases, with varying scales and complexity:
- Kelvin-Helmholtz instability (KHI) - a 2D flow instability with vortices (and in our example, wave shocks).
- Massachussetts Bay (MassBay) ocean data - one month of real ocean flow with complex, multiscale features.

In all cases, we provide the $u,v$ 2D velocity fields as data, and compute vorticity for visualization. All data is normalized to $[-1,1]$.

### Linear Channel Compressor [LCC]
We first define a tokenizing layer that adds a linear positional encoding and increases/decreases the number of channels. This layer is pointwise on the tokens (which are pixels in this work), and is thus linear in x-y position and computationally linear in the number of pixels / tokens.

This layer replaces the traditional linear fully-connected layer (but also loses a lot of nonlinearity).

### Convolution Layer
We define convolutions with fixed kernel size 5x5 and stride 1; the channels are similarly fixed to avoid any inductive bias regarding scale.

Of course, the kernel size itself is not scale-invariant, but is the closest we can get to a tractable scale-invariant convolution.
If trained appropriately with varying-resolution input, the positional encoding should be able to enforce scale-invariance.

We leave scale-invariant convolutions for future work, so *do not* yet train on varying-resolution data.
Similarly, to keep the architecture as simple and interpretable as possible, we do not use any transposed convolutions.**

### Overall Architecture
We train the main autoencoder as below with a latent space of 240 dimensions for all cases (arbitrarily chosen). Post-training, we train a fully-connected ReLU autoencoder to compress the latent space down to the latent-space intrinsic dimension.

Note architectures without convolution were tested, and the addition of convolutions somewhat improved performance by collecting local information.

<img src="plot/arch.svg"></img>

For the Kelvin-Helmholtz instability (KHI) dataset, we switch out the LCC-Conv compressor for a 3x iterate of the same compressor layer. This emulates a fixed-point iteration layer to help a wave-like solution.

The tests on the full, Massachussetts Bay (MassBay) ocean data are done with a latent dimension of 120, and the non-fixed-point compressor. This is primarily due to compute constraints.
Some incomplete tests were done at 240 and / or deeper networks, but these failed to converge.

We share some small ablation studies below; these are *not exhaustive by intention*.

#### Activation Test
As might be obvious, sigmoid or tanh activations are likely not appropriate, since there is no clear probability distribution in the data. We test ReLU against Sine activations on the MassBay dataset, and find that Sine activations are visually better at capturing the diverse wave shapes.

<div class='alert'>
Tip: Most reconstructions are fluid flow videos; hover over the plot to display controls.
</div>
<object id='plot' data-src="MB/Act/relu/valid.mp4" title="MB, relu activation"></object>
<object id='plot' data-src="MB/Act/sine/valid.mp4" title="MB, sine activation"></object>
<details>
<summary>Inference</summary>
Above is validation, below is inference.

<object id='plot' data-src="MB/Act/relu/infer.mp4" title="MB, relu activation"></object>
<object id='plot' data-src="MB/Act/sine/infer.mp4" title="MB, sine activation"></object>
</details>

#### Fixed-Point Comparison
Fixed-point vs the standard compressor: clearly fixed-point is better for this dataset; notice the sharper wave-like features, esp. at 0:02.

<object id='plot' data-src="KHI/FP/valid.mp4" title="KHI with Fixed-Point Architecture"></object>
<object id='plot' data-src="KHI/FP/control/valid.mp4" title="KHI with standard architecture"></object>

<details>
<summary>Inference</summary>
Above is validation, below is inference.

<object id='plot' data-src="KHI/FP/infer.mp4" title="KHI with Fixed-Point Architecture"></object>
<object id='plot' data-src="KHI/FP/control/infer.mp4" title="KHI with standard architecture"></object>
</details>


### EMD Loss
We use an Earth-Mover-like loss to enforce position and value similarity, to avoid blurry issues with L2 solutions.
The L2 distance is upgraded to the graph space using a local 10x10 stencil ($k=10$, $\gamma=0.25$).

$d_{EMD}^2 = \min_{i \in [-k,k]} \min_{j \in [-k,k]} {d(\hat{y}_{ij},y)}^2 + \gamma {d(x_i,x)}^2 + \gamma {d(y_j,y)}^2$

Below are some results for the MassBay dataset, with kernel sizes 5x5 and 10x10; comparing against the previous Sine experiment (with standard L2), we find the 10x10 EMD best.
The exact kernel size is not critical as we are not chasing performance yet (and it will depend on data).

<object id='plot' data-src="MB/EMD/k5/valid.mp4" title="EMD with 5x5 kernel"></object>
<object id='plot' data-src="MB/EMD/k10/valid.mp4" title="EMD with 10x10 kernel"></object>

<details>
<summary>Inference</summary>
Above is validation, below is inference.

<object id='plot' data-src="MB/EMD/k5/infer.mp4" title="EMD with 5x5 kernel"></object>
<object id='plot' data-src="MB/EMD/k10/infer.mp4" title="EMD with 10x10 kernel"></object>
</details>

## Kelvin-Helmholtz Instability (KHI) Results

### Training

Interestingly, despite the entropy measure being about 1% of the total training loss, it still freezes the training process. We speculate a large gradient is produced by the entropy term, which is easily minimized but heavily constrains reconstruction.
Hyperparameter tuning was not done, since the measure is still arbitrary.

Note the red curve is unconstrained, and the orange curve is with the entropy regularization.

<img src="plot/KHI/train/train_loss.png" width="32%"></img>
<img src="plot/KHI/train/valid_loss.png" width="32%"></img>
<img src="plot/KHI/train/latent_entropy.png" width="32%"></img>

Note the model might not be fully trained, but any futher is infeasible based on available compute (2 A6000 GPUs).
Note also the model is not exactly memorizing, as the train loss is not 0; it is rather stagnating.
The parameter gradients corroborate.

[with entropy regularization]

<img src="plot/KHI/train/entropy_weights.png" width="80%"></img>

Notice for the both models, the shown encoder layers (first two columns) flatlines initially, while the first decoder layer (last column) starts learning immediately.
This is essentially the network creating the gradient flow; until the unconditional probability distribution of the decoder is learned, the encoder cannot learn.

But the non-entropy model starts learning the encoder layers much earlier:

<img src="plot/KHI/train/_weights.png" width="80%"></img>

### Reconstruction
The MSE loss is quite close, so the models both perform close.

Clearly, though, the wave-like features are not captured exactly. As expected, the constrained network (left) has worse performance.

<object id='plot' data-src="KHI/Info/valid.mp4" title="KHI with minimum entropy regularization"></object>
<object id='plot' data-src="KHI/Control/valid.mp4" title="KHI without minimum entropy regularization"></object>

<details>
<summary>Inference</summary>
Above is validation, below is inference.

<object id='plot' data-src="KHI/Info/infer.mp4" title="KHI with minimum entropy regularization"></object>
<object id='plot' data-src="KHI/Control/infer.mp4" title="KHI without minimum entropy regularization"></object>
</details>

It does capture some large-scale shocks, but not fine; this matches the intrinsic dimension measurements.
It follows that these wave features nonlinearly depend on scale (else the network would easily learn them via the positional encoding).
Physically this makes sense. Waves are 2nd-order advective solutions, and these caustics are accumulations of multiple reflections and refractions---which are not easily captured by quasi-linear networks, even when using a sinusoidal basis (sine activation).

### Intrinsic Dimension

Computing the intrinsic dimension for data, latent space, reconstruction, and error, we find additive agreement (given an uncertainty of at least +-1).

> Data: $\in$ [6,8] dims  
> Latent $\in$ [2,2.5] dims  
> Reconstruction $\in$ [3,3.5] dims (greater due to noise production)  
> Error: $\in$ [3.5,4] dims (with the notable exception of the training dataset...)  

So the network loses about 1/2 of the information (which is in the pathological caustics)!
With the min-entropy regularization:

> Latent = 1.5 dims  
> Reconstruction $\in$ [1.5,2] dims (slightly greater due to noise)  
> Error: $\in$ [3,4.5] dims

So this network loses significantly more information!
Notice the MSE loss from training is not that different, so the mean/variance/low-order statistical moments were preserved.

For exact values / evidence / uncertainty and dataset variations, we link the corresponding Two-NN volume <a href="plot/KHI/KHI-ID-scale-invariants.pdf">ratio plots and fits</a>.

### Limiting Sequences in Intrinsic Dimension Space
So how does this affect the sequences?
We plot the [control] on the left with 3 intrinsic dimensions vs the [min-entropy-regularization] on the right with 2 intrinsic dimensions.

<div class='alert'> Note: Page zoom affects the plots quite a bit, so we appreciate your understanding.
Please adjust the sliders as needed to see single tracks or the full manifold.
</div>

<iframe src='plot/KHI/Control/train_latent_seq.html' id='emb'></iframe>
<iframe src='plot/KHI/Info/train_latent_seq.html' id='emb'></iframe>
<iframe src='plot/KHI/Control/valid_latent_seq.html' id='emb'></iframe>
<iframe src='plot/KHI/Info/valid_latent_seq.html' id='emb'></iframe>
<iframe src='plot/KHI/Control/infer_latent_seq.html' id='emb'></iframe>
<iframe src='plot/KHI/Info/infer_latent_seq.html' id='emb'></iframe>

Interestingly, the control network has a complex training visualization; this is the exception from the intrinsic dimension results as well; likely the regularization minimizes train-test variation, while the control network generalizes better and just ignores the hard-to-learn train-data features.

The entropy regularization has clearly killed all the dimensions to essentially one; the sequences are now hard to distinguish, but some likely still exist.
The restriction of the true sequence should be somewhat visible unless the network has diverged (though again it is hard to tell).

The control network has a clear separation of the sequences, with almost linear tracks in some cases. The overall manifold is much more complex, though, and there isn't an obvious "high-resolution" area. As the intrisic space (as a tangent space) is unique up to a rotation (and reflection depending on orientability of the manifold), the local coordinate charts (tangent space encodings) may not be be consistent, even if continuous on the entire manifold.
This is especially true for networks with ReLU / activations with disjoint regions.

We expect a simple transformer network to easily "superresolve" data on these linear tracks for the control network.
While a higher-dimensional latent space with
- a more linear manifold
- uniquely defined via min-entropy-regularization

may be better for forecasting, it is clearly not necessary for superresolution (in the intrinsic-dimension tangent space).


Further research will investigate forecasting to see if the intrinsic space is sufficient.

### Scale-Invariant Features

Towards building a scale-invariant forecasting model, what scale-invariant features did we extract?

We plot scans along each latent-space axis for training, validation, and inference data (5 steps each) for the control and entropy-regularized models respectively.
Scans exceed the natural data range, so as to exaggerate the features; we don't need to worry about seams as the latent manifold is not directly constrained (as in a VAE).

<object id='plot2' data-src="KHI/Control/latent_space_scan_0.png" title="Control Latent Scan"></object>

This axis seems to indicate spread of the central flow; the images go from wide vertical flow to narrow.

<object id='plot2' data-src="KHI/Control/latent_space_scan_1.png" title="Control Latent Scan"></object>

This axis seems to indicate a sort of handedness; the images go from left-protrusions in flow to right-protrusions.

<object id='plot2' data-src="KHI/Control/latent_space_scan_2.png" title="Control Latent Scan"></object>

This axis seems to indicate a sort of magnitude; the images go from strong vertical and horizontal flow to weak.

Recall the entropy-regularized latent space is only 2D, so it is more difficult to interpret.

<object id='plot2' data-src="KHI/Info/latent_space_scan_0.png" title="Entropy Latent Scan"></object>

This axis seems like a combination of the magnitude and flow width (natural to go together, even if not necessarily physically meaningful).

<object id='plot2' data-src="KHI/Info/latent_space_scan_1.png" title="Entropy Latent Scan"></object>

This axis seems like a noisy version of the handedness axis.

We will connect these features to fluid operators in future work, once the network produces sharper and more easily identifiable features.

### Future Work
We have shown that a scale-invariant autoencoder is possible at the intrinsic-dimension limit, if not yet at full image recovery.
Also, we find regularizing such a highly-compressible model might easily lose important information, and compression should be preferred over regularization.

Further work will investigate:
- Full image recovery
- Encoding for pathologies like caustics
- Increased training and lower model complexity
- Operator identification
- Actual (MassBay) ocean data at the intrinsic-dimension limit
- Forecasting with the intrinsic dimension

## notes / disclaimers

A flow past a cylinder (FPC) - a simple, low-resolution, 2D CFD (computational fluid dynamics) benchmark was also tested but did not converge since training data covered a phase-transition.
Retraining is ongoing.

All plots are individually normalized to max/min of ground-truth data. $\omega$ denotes a vorticity formulation (which is only accurate to literature for the not-shown flow-past-a-cylinder case, since it depends on sign convention).

Nearly all tests/experiments in this work are qualitative, and are *not intended* to be quantitative (yet). This is due to time-constraints and since the model is likely not fully trained.
This work is in no way a full investigation, but rather a proof-of-concept.

*Some current research states that 2 snapshots of memory is sufficient for weather forecasting, but this is not the case for all problems. Furthermore, that research did not fully train models (only 10 epochs) or have capability to test much larger sequences, so their conclusions are suspect.

**Technically the decoder is not an inverse of the encoder but a separate network. This is true both with transposed convolution and with the convolution-only approach.
Do note that to invert a convolution (which is a local, sparse matrix), a global, dense matrix is required. This is not easily made scale-invariant or tractable, and requires further attention.
One could argue that transposed convolutions are a better approximation.

***A better approach (which we will try next) would be to maximize the entropy and minimize the number of active variables. The entropy maximization naturally leads to an optimal compressor/encoder on the variables (see MIT 6.7480 HW.9 P.6). While entropy minimization as done in this work does minimize the total description, likely the objective is so opposite the optimal compressor objective that the encoder fails.

## references
citation{SINDy, [Discovering governing equations from data: Sparse identification of nonlinear dynamical systems](https://arxiv.org/abs/1509.03580)}
citation{SpacetimePOD, [Space-time POD and the Hankel matrix](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289637)}
citation{Attention, [Attention is All You Need](https://arxiv.org/abs/1706.03762)}
citation{FuXi, [FuXi: A cascade machine learning forecasting system for 15-day global weather forecast](https://arxiv.org/abs/2306.12873)}
citation{ClimODE, [ClimODE: Climate and Weather Forecasting with Physics-informed Neural ODEs](https://arxiv.org/abs/2404.10024)}
citation{NeuralGCM, [Neural general circulation models for weather and climate](https://www.nature.com/articles/s41586-024-07744-y)}
citation{FNO, [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)}
citation{AFNO, [Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers](https://arxiv.org/abs/2111.13587)}
citation{SFNO, [Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere](https://arxiv.org/abs/2306.03838)}
citation{UNet, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)}
citation{ViT, [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)}
citation{Diffusion, [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)}
citation{VAE, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)}
citation{TwoNN, [Estimating the intrinsic dimension of datasets by a minimal neighborhood information](https://arxiv.org/abs/1803.06992)}
citation{PACBayesGeneralize, [Non-Vacuous Generalization Bounds at the ImageNet Scale: A PAC-Bayesian Compression Approach](https://arxiv.org/abs/1804.05862)}
citation{InformationTheory, [Information Theory: From Coding to Learning](https://people.lids.mit.edu/yp/homepage/data/itbook-export.pdf)}-->
`; 
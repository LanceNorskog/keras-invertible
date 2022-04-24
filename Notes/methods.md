# Simple Invertible Layers
## Dense
Inverse of Dense is transpose of Dense, with activation->bias->transpose.

## Activations

Some activations are invertible, and in fact has inversions available. 
Sigmoid/Logit, Tanh/Arctanh. RELU is not invertible.

### Swish

https://www.tensorflow.org/api_docs/python/tf/keras/activations/swish

Swish is x * sigmoid(x), so inverse is logit(x)/x

Of course, there are places in Swish output which "cross", meaning that if we wish to invert a Swish layer,
there are two possible outputs for at least one input. 
But we will ignore this, hoping that this will wash out in real use.
Also, in a real model, we will need to train replacement layers (see below) and that will help cover up the problem.

## Convolutions
A Convolutional layer is an ordered collection of small Dense kernels. 
These can be inverted by transposing the learned kernels. 
The "Transpose" set of CNN layers was designed for this inversion.

## Concatenate
Concat -> something is inverted by slices

# Inverting via Learnable Replacements

## Merge Layers
Merge Layers (besides Concatenate) are (mostly) non-invertible. It should be possible to approximate these via learned values. These merges must be replaced with a learnable invertible replacement, and learn weights based on that replacement. Then, the matching inverted set of layers forms an approximate inversion of the original layer.

Replacing in a learned model (residual signal technique for example) must freeze the original model, leave the replacement layer trainable, then train the replacement layers to respond appropriately for the context of the dataset.
### Note
This design bets that the multiple inputs to a function have a small standard deviation in each input neuron. 
That is, position 43 of an Add(input A, input B) usually has a large value on input A and a small value on input B.

If inputs have a high standard deviation, and we want to replicate this, we probably need to add a matching amount of noise.

### Add
F.x. Add can be replaced with "Concat -> Dense(no bias, no activation)" and trained. Output is (transpose Dense, slices)

Or, for residual signal maybe have a Dense the size of one input, and add one input and subtract another input, where both are multiplied by the Dense.
That is: (Add(feature map, residual) becomes Add(feature map * Dense, residual * -Dense), and invert becomes 2 outputs(Sum * Dense, Sum * -Dense).

This design bets that the signal level contributed by the residual input is usually smaller than the signal contributed by the feature map.

# Layer Norms

## BatchNorm
BatchNorm is a learned transform, and I think the rescaling is invertible. Have to check though.

## Layer Norms
Other layer norms are not learned, and need to be replaced with learned xforms. It might be that a Dense layer with N weights per input is easy and works.

### Log(input - mean)
If Layer norm is log(input subtract mean), then the replacement is (Dense, bias, with activation log) and the inverse is (exp -> -bias -> transpose Dense). 

# Pooling
## Global Average Pooling 
This is a hard one. If it has a low standard deviation, it can be replaced by fanning out the "output" value to all inputs. If the GAP layer has a high standard deviation (which it probably does) it could be inverted by fanout * noise. For the latter, this would require a training phase to acquire the standard deviation. 

## Global Max Pooling
I doubt it is legitimate, but a noise replacement technique might get better results here than just fanning out the max to all inputs.

# Misc tools
## Bias Layer
Separate layer that uses weights from a Dense layer. Can add or subtract.
## Noise Reconstitution Layer
If an input signal has high standard deviation, and the signal is smoothed in some way, we may want to recreate the standard deviation via adding noise to the inversion. 

Example: Noise -> Global Average Pooling, where Inversion of GAP does simple fan-out.

# Rebuilding pretrained models
It should be possible to rebuild an existing trained network by replacing all of the existing layers with either inverting layers or the above rebuilding techniques,
if the model is amenable. The EfficientNet series of models are (I think) possible to rebuild this way. 
EfficientNet models have ImageNet weights available, and it should be possible to reverse one of these models.
This would create a new way to explore the weights of an ImageNet-based model.

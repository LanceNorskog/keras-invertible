# Dense
Inverse of Dense is transpose of Dense, with activation->bias->transpose.

# Bias
Inverse of Bias needs (input - weights)

# Activations

Some activations are invertible, and in fact has inversions available. 
Sigmoid/Logit, Tanh/Arctanh. RELU is not invertible.

### Swish

https://www.tensorflow.org/api_docs/python/tf/keras/activations/swish

Swish is x * sigmoid(x), so inverse is logit(x)/x

Of course, there are places in Swish output which "cross", meaning that if we wish to invert a Swish layer, \
there are two possible output for at least one input. 
But we will ignore this, hoping that this will wash out in real use.
Also, in a real model, we will need to train replacement layers (see below) and this oddity will probably wash out.

# Convolutions
A Convolutional layer is an ordered collection of small Dense kernels. 
These can be inverted by transposing the learned kernels. 
The "Transpose" set of CNN layers was designed for this inversion.

# Learnable inversions

## Merge Layers
Merge Layers are (mostly) non-invertible, but can be approximated via learned values.

Concat -> something is replaced by slices
That is, Concat -> Dense can be replaced with inverse Dense -> slices

Other merges must be replaced with a learnable invertible replacement, and learn weights based on that replacement.
Replacing in a learned model (residual signal technique for example) must freeze the original model, then train the replacement.
This design bets that the multiple inputs to a function have a small standard deviation in each input neuron. 
That is, position 43 of an Add(input A, input B) usually has a large value on input A and a small value on input B.

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

# Rebuilding pretrained models
It should be possible to rebuild an existing trained network by replacing all of the existing layers with either inverting layers or the above rebuilding techniques,
if the model is amenable. The EfficientNet series of models are (I think) possible to rebuild this way. 
EfficientNet models have ImageNet weights available, and it should be possible to reverse one of these models.
This would create a new way to explore the weights of an ImageNet-based model.

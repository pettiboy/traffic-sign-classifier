# Traffic

### Getting Started

I started by experimenting with the example from the lecture having one Convolutional layer and one Pooling layer.

### Incredibly high loss

I tried multiple times by changing number of filters in the `Convolutional layer` experimenting with different `pool sizes` and varing the
nodes and number of `hidden layers` but my `accuracy` was always less that` 0.85` and by `loss` was sometimes `100+`
I was unable to understand why that was the case...

### Problem persists

I kept trying by increasing the number of `hidden layers`, changing `activation functions` and tried changing the `Dropout rate`.
But the problem still persisted, I had less that `0.90 accuracy` and incredibly `high loss (100+)`.
There were times when with each epoch the `loss kept increasing`..

### Removing dropout

After that I tried to remove the Dropout completely to see what happens. To my surprise the model did not overfit and for the first time I got `loss less than 1`!!

### Experimenting without dropout

Now with loss less than one, I further tried changing number of `Convolutional and Pooling layers`
until I finally reached `accuracy greater than 0.95`

### Preprocessing data by rescaling

After going through tensorflow documentation (https://www.tensorflow.org/tutorials/images/classification) I discovered `rescaling`
which would `standardize values` to be in the [0, 1] range.
This made data `easier to work` with for the neural network and resulted in `accuracy of over 0.98` and loss less that 0.09!

### Adding strides in Pooling

Now I tried experimenting with various option in tensorflow api docs (https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).
Adding strides in pooling futher increased my accuracy but increased the execution time. Doubling number of Convolutional filters also
increased efficiency marginally.

### Trying increasing number of activation layers

I further tried increasing the number of activation layers but it did not increase accuracy.

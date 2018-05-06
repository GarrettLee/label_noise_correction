# label_noise_correction
Implementation of paper: Making Deep Neural Network Robust to Label Noise: a Loss Correction Approach.

## Requirements:
- Python 2.7
- TensorFlow 1.4
- Matplotlb
- Numpy

## Usage
- Train all models and evaluate all the tests with: `python experiment_mnist.py`, or with `bash run_experiment_mnist` for faster training and testing. When this is finished, 4 files named `backward.npy`, `backward_t.npy`, `cross_entropy.npy`, `forward.npy`, `forward_t.npy` should have been created under the path `./result/mnist/`.
- Show the result with: `python show_result_of_mnist_experiment.py`.

## Result
This is the result of *Fully connected network on MNIST*. Notice that when N=0.5, the parametric matrix T is singular.
![results.png](https://raw.githubusercontent.com/GarrettLee/label_noise_correction/master/results.png)

# Neural Network MNIST

A handwritten digit recognition neural network written completely from scratch in Python.

The project focuses on understanding and implementing the mathematical foundations of neural networks without using machine learning frameworks such as TensorFlow or PyTorch.

---

## Features

* Feedforward neural network
* Backpropagation
* Sigmoid activation function
* MNIST handwritten digit recognition
* Configurable:

  * hidden nodes
  * learning rate
  * epochs
* Live testing application
* Performance evaluation and parameter analysis

---

## Technologies

* Python 3
* NumPy
* SciPy
* Matplotlib
* Tkinter

---

## Project Structure

```txt
.
├── Live_testing.py
├── docs/
├── mnist_test.csv
├── weights_hidden_output.txt
└── weights_input_hidden.txt
```

---

## Mathematical Background

The neural network uses:

* Matrix multiplication
* Sigmoid activation
* Gradient descent
* Backpropagation

The project also evaluates the influence of:

* hidden nodes
* learning rate
* epochs

on the network accuracy.

---

## Results

The best tested configurations achieved an accuracy of approximately:

```txt
97.6%
```

on the MNIST test dataset.

---

## Example

Example digit from the MNIST dataset:

![Example](docs/example_digit.png)

---

## Documentation

Additional mathematical explanation and evaluation can be found in:

* [Documentation PDF](docs/documentation.pdf)

---

## Goal of the Project

The purpose of this project was to better understand:

* how neural networks work internally
* how training and backpropagation function mathematically
* how parameter tuning affects performance

The neural network was intentionally implemented without high-level AI frameworks.

---

## Author

Mathis Grün

# Experiments

These experiments are design to highlight how key features in deep learning behave.

- [test_graph](test_graph.py) checks that the computational graph behaves as expected on a simple feed forward network.
- [test_sgd](test_sgd.py) comparison of the most popular optimization algorithms for a deep feed-forward network fitted on the ``make_classification`` problem of Scikit-learn.
- [test_dropout](test_dropout.py) fits a deep neural net to the MNIST dataset. The experiment compares performance with and without dropout. Dropout outperforms all Scikit-learn benchmarks.
 

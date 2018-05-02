# Regression on non-linear data [![Build Status](https://travis-ci.com/kqf/time-series-regression.svg?branch=master)](https://travis-ci.com/kqf/time-series-regression)

This is the solution of regression problem on non-linear dataset that was measured in different time intervals.

## The solution
The most optimal solution relies on SVM.

To run the solution do
```bash

# Download the dataset *.csv into ./data folder 
# and then run


# It will train and validate the model for the
# best parameters I was able to find
make 
```


## Steps

1. Explore the data
2. Read/process the data
3. Model complexity (try simple models first):
	- Linear models (Linear, L1, L2, ElasticNet). The problem is nonlinear. Polynomial features (slow)
	- Try generalized linear models SVR for different kernels. Tune parameters.
	- Look if we can benefit from from complex models like Random Forests
	- Try to improve the result with neural networks
4. Tune parameters (almost done)
6. Further improvemets:

## Results
TODO: Add finetuned results
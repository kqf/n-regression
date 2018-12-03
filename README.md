# Regression on non-linear data [![Build Status](https://travis-ci.com/kqf/time-series-regression.svg?branch=master)](https://travis-ci.com/kqf/time-series-regression)

This is a toy-regression problem on non-linear dataset that was measured in different time intervals. The actual dataset is not public and therefore test fixtures create fake data files with the same schema, so the code can be tested. The model parameters are minimal so tests can run faster.

## Run the solution
The most optimal solution relies on SVM.

To run the solution do
```bash

# Download the dataset *.csv into ./data folder 
# and then run


# It will train and validate the model for the
# best parameters I was able to find
make 
```

# Regression on non-linear data

This is the solution of regression problem on non-linear dataset that was measured in different time intervals.



## The solution
My solution uses Random Forest Regression to achieve the best results on training data.


To run the solution do

```bash

# Download your dataset *.csv into ./data folder 
# and then run


# It will train and validate the model for the
# best parameters I was able to find

make 
```


## Steps
Here is a list of steps that I followed to get the results

1. Look at data (see jpyter notebook), to find (if any) correlated variables, determine the most important features.
2. Read/process the data
3. Try linear model and compare to RandomForest
4. Tune parameters (almost done)
5. Use the best parameters as to get the best solution (didn't do this yet)
6. Further improvemets:
    - Simpler models
    - L1 regularization to select features
    - Tweak the most/least informative feature to see the impact on the MSE
    - Does this really depend on time?

## Results
The scores on the best parameters (that I've found so far)
```
Random Forest MAE 241.093120355
Random Forest MSE 141806.236619

```

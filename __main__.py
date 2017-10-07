from model.training import Trainer as ttr 
from sklearn.ensemble import RandomForestRegressor

def main():
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, criterion='mse', random_state = 1)
    ttr.check_model('Random Forest', rf)

if __name__ == '__main__':
    main()


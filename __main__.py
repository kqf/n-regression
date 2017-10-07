from model.training import Trainer as ttr 
from sklearn.ensemble import RandomForestRegressor

def main():
    ttr.check_model('Random forest', RandomForestRegressor())

if __name__ == '__main__':
    main()


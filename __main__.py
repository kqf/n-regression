from model.training import Trainer as ttr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def main():
    rf = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=25),
    )
    ttr.check_model("SVR", rf)


if __name__ == "__main__":
    main()

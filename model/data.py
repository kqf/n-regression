import pandas as pd

# Use object to be able to override it later


class DataHandler():

    @classmethod
    def _load_files(klass, ifile='data/inputs.csv', tfile='data/targets.csv'):
        return pd.read_csv(ifile), pd.read_csv(tfile)

    @classmethod
    def _time_data(klass, data, target):
        data['timeStamp'] = data.timeStamp.apply(pd.to_datetime)
        data['month'] = data.timeStamp.apply(lambda x: x.month)
        data['hour'] = data.timeStamp.apply(lambda x: x.hour)
        # TODO: add woringdays and holidays?
        data['target'] = target
        return data

    @classmethod
    def load(klass, ifile='data/inputs.csv',
             tfile='data/targets.csv'):
        data, target = klass._load_files(ifile, tfile)
        assert not data.isnull().values.any(), \
            'Your data has null entries, clean it'
        data = klass._time_data(data, target)
        return data

    @classmethod
    def load_train_test(klass, ifile='data/inputs.csv',
                        tfile='data/targets.csv', fraction=0.25):
        data = klass.load(ifile, tfile)
        testsize = int(data.shape[0] * fraction)
        train = data.iloc[:-testsize, :]
        test = data.iloc[-testsize:, :]

        assert train.timeStamp.max() < test.timeStamp.min(), \
            'Problems with your data,' \
            'make sure that you are trying to predict future'

        # Drop the datetime
        train = train.drop(['timeStamp'], axis=1)
        test = test.drop(['timeStamp'], axis=1)

        return (train.drop(['target'], axis=1), train.target.values),\
               (test.drop(['target'], axis=1), test.target.values)

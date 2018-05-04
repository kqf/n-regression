import pandas as pd


def tofloat(df):
    for colname in df.columns:
        try:
            df[colname] = df[colname].apply(float)
        except TypeError:
            pass
    return df


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
        fdata = tofloat(data)
        testsize = int(fdata.shape[0] * fraction)
        train = fdata.iloc[:-testsize, :]
        test = fdata.iloc[-testsize:, :]

        assert train.timeStamp.max() < test.timeStamp.min(), \
            'Problems with your data,' \
            'make sure that you are trying to predict future'

        # Drop the datetime
        train = train.drop(['timeStamp'], axis=1)
        test = test.drop(['timeStamp'], axis=1)

        return (
            train.drop(['target'], axis=1),
            test.drop(['target'], axis=1),
            train.target.values,
            test.target.values,
        )

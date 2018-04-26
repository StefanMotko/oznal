import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.grid_search import GridSearchCV

np.random.seed(123456789)

class ColumnGetter(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, data, target=None):
        return self

    def transform(self, data):
        return data.iloc[:,self.columns]


class LinearApproximator(BaseEstimator, TransformerMixin):

    def fit(self, data, target=None):
        return self

    def transform(self, data):

        def approximate(row):
            return np.polyfit(range(0, row.size), row, 1)[0]

        return pd.DataFrame(data.apply(approximate, axis=1, reduce=True))


columnSpecificTransforms = FeatureUnion([
    ('DropFirstCorrelatedSquare', Pipeline(steps=[
        ('ColumnGetter', ColumnGetter([5])), # ignore 6, 7, 8, 9, 10
    ])),
    ('PCASecondSquare', Pipeline(steps=[
        ('ColumnGetter', ColumnGetter([11, 12, 13, 14, 15, 16])),
        ('PCA', PCA()),
        ('QuantileTransform', QuantileTransformer(output_distribution='normal')),
        ('Rescale', StandardScaler())
    ])),
    ('RescaleLongTails', Pipeline(steps=[
        ('ColumnGetter', ColumnGetter([1, 17, 18, 19, 20, 21, 22])),
        ('QuantileTransform', QuantileTransformer(output_distribution='normal')),
        ('Rescale', StandardScaler())
    ])),
    ('LinearApproximation', Pipeline(steps=[
        ('ColumnGetter', ColumnGetter([11, 12, 13, 14, 15, 16])),
        ('Approximator', LinearApproximator())

    ])),
    ('NonModified', ColumnGetter([2, 3, 4]))
])

pipeline = Pipeline(
    steps=[
        ("ColumnSpecificTransforms", columnSpecificTransforms),
        ('Classifier', AdaBoostClassifier(ExtraTreesClassifier()))
    ]
)

param_grid = {
    'Classifier__base_estimator__n_estimators': [ 10, 50, 100 ],
    'Classifier__base_estimator__criterion': [ 'gini', 'entropy' ],
    'Classifier__base_estimator__min_samples_leaf': [ 1, 50, 500 ]
}

print('Created pipeline')

data = pd.read_csv('credit_card_default.csv')

data_0 = data[data['DEFAULT_PAY'] == 0]
data_1 = data[data['DEFAULT_PAY'] == 1]

train_0, test_0 = train_test_split(data_0, train_size=0.75)
train_1, test_1 = train_test_split(data_1, train_size=0.75)

train_original = pd.concat([train_0, train_1])
test_original = pd.concat([test_0, test_1])

# print('>>>> Preserved ratio')
data_1 = data[data['DEFAULT_PAY'] == 1]

train_0, test_0 = train_test_split(data_0, train_size=0.75)
train_1, test_1 = train_test_split(data_1, train_size=0.75)

train_original = pd.concat([train_0, train_1])
test_original = pd.concat([test_0, test_1])

# print('>>>> Preserved ratio')
#
# target = train_original['DEFAULT_PAY']
# train = train_original.drop(labels='DEFAULT_PAY', axis=1, inplace=False)
# test = test_original.drop(labels='DEFAULT_PAY', axis=1, inplace=False)
#
# print('     Loaded data')
#
# pipeline.fit(train, target)
#
# print('     Fitted pipeline')
#
# result = pipeline.predict(test)
#
# print('     Predictions complete')
#
# pd.DataFrame(test_original['DEFAULT_PAY']).to_csv("target_pres.csv", index=False, header=False)
# pd.DataFrame(result).to_csv("predict_pres.csv", index=False, header=False)

print('>>>> Ratio normalization by under-sampling')

train = pd.concat([
    train_0.sample(n=7500),
    train_1
])
target = train['DEFAULT_PAY']
test = test_original.drop(labels='DEFAULT_PAY', axis=1, inplace=False)

train.drop(labels='DEFAULT_PAY', axis=1, inplace=True)

print('     Loaded data')

# pipeline.fit(train, target)
search = GridSearchCV(pipeline, param_grid, scoring='f1_micro')
search.fit(train, target)
print(search.best_params_)

# print('     Fitted pipeline')

# result = pipeline.predict(test)

# print('     Predictions complete')

# pd.DataFrame(test_original['DEFAULT_PAY']).to_csv("target_under.csv", index=False, header=False)
# pd.DataFrame(result).to_csv("predict_under.csv", index=False, header=False)

# print('>>>> Ratio normalization by over-sampling')
#
# train = pd.concat([
#     train_0,
#     train_1,
#     train_1,
#     train_1
# ])
# target = train['DEFAULT_PAY']
# test = test_original.drop(labels='DEFAULT_PAY', axis=1, inplace=False)
#
# train.drop(labels='DEFAULT_PAY', axis=1, inplace=True)
#
# print('     Loaded data')
#
# pipeline.fit(train, target)
#
# print('     Fitted pipeline')
#
# result = pipeline.predict(test)
#
# print('     Predictions complete')
#
# pd.DataFrame(test_original['DEFAULT_PAY']).to_csv("target_over.csv", index=False, header=False)
# pd.DataFrame(result).to_csv("predict_over.csv", index=False, header=False)





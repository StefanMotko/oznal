import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

dtc = DecisionTreeClassifier()

pipeline = Pipeline(
    steps=[
        # ('DecisionTree', dtc)
        # ('MultinomialNB', MultinomialNB())
        ('LogisticRegression', LogisticRegression())
    ]
)

print('Created pipeline')

data = pd.read_csv('credit_card_default.csv')

target = data['DEFAULT_PAY']
data.drop(labels='DEFAULT_PAY', axis=1, inplace=True)

print('Loaded data')

pipeline.fit(data, target)

print('Fitted pipeline')

result = pipeline.predict(data)

print('Predictions complete')

pd.DataFrame(result).to_csv("predict.csv", index=False, header=False)
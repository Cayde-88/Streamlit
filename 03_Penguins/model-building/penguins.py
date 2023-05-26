import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
penguins = pd.read_csv('./penguins_cleaned.csv')
target = 'species'
encode = ['sex', 'island']

# get dummy data
for col in encode:
    dummy = pd.get_dummies(penguins[col], prefix=col)
    penguins = pd.concat([penguins, dummy], axis=1)
    del penguins[col]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

penguins['species'] = penguins['species'].apply(target_encode)

# Split X and y
X = penguins.drop('species', axis=1)
Y = penguins['species']

# Build Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Save the model
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
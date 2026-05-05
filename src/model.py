import pickle
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
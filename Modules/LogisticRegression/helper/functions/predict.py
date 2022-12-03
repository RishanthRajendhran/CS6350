from Modules.helper.imports.packageImports import np

def predict(weights, X):
    augX = np.ones((X.shape[0], X.shape[1]+1))
    augX[:,:-1] = X.copy()

    X = augX.copy()
    return np.sign(np.dot(X, weights))
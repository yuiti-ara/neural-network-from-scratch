from neural_net import NeuralNet


if __name__ == '__main__':

    import numpy as np
    X_tr = np.random.randn(2, 3)
    X_ts = np.random.randn(2, 3)
    Y_tr = np.random.randn(1, 3) > 0

    model = NeuralNet(debug=True)
    model.fit(X_tr, Y_tr)

    Y_pred = model.predict(X_ts)
    print(Y_pred)

import pytest
import numpy as np

from neural_net import sigmoid, fit_params

np.random.seed(1)


class TestFunctions:

    in_out_sigmoid = [
        (0, .5),
        (1, .731058),
        (-1, .268941),
        (10, .99995),
        (-10, .00004539)
    ]

    @pytest.mark.parametrize('x, y', in_out_sigmoid)
    def test_sigmoid(self, x, y):
        assert np.isclose(sigmoid(x), y)


class TestModel:

    def test_fit_params(self):

        X = np.array([[+1.62434536, -0.61175641, -0.52817175],
                      [-1.07296862,  0.86540763, -2.30153870]])
        Y = np.array([[True, False, True]])

        params_expected = {
            'W1': np.array([[+0.72222553, -1.32296487],
                            [+0.60340600, -1.13482071],
                            [+0.80358022, -1.47814908],
                            [-0.63389190,  1.19106035]]),
            'b1': np.array([[-0.32851206],
                            [-0.25706940],
                            [-0.38114549],
                            [+0.27762883]]),
            'W2': np.array([[+2.92155935,  2.12503828,  3.72368559, -2.33235831]]),
            'b2': np.array([[+0.20638534]])
        }

        # test function
        params = fit_params(X, Y, 4)
        for key, param in params.items():
            assert np.allclose(param, params_expected[key])

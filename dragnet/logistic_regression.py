
import numpy as np
from scipy.optimize import fmin_bfgs
import json

class LogisticRegression(object):
    def __init__(self, lam=1.0):
        """lam = regularization parameter"""
        self.lam = lam

        # these are set in learn
        self.b = None  # float
        self.w = None  # (nvars, ) array

    def pred(self, x):
        """Make a prediction.
        Return P(y == 1 | x)

        x = (Nobs, nvars)
        """
        return LogisticRegression._sigmoid(x, self.b, self.w)

    def learn(self, x, y, weights=None):
        """Train the model.

        x = (Nobs, nvars)
        y = (Nobs, )  = {0, 1}

        Bias term automatically added

        Returns the loss"""

        y0 = y == 0
        x0 = x[y0, :]
        x1 = x[~y0, :]

        if weights is None:
            loss_weights = None
        else:
            loss_weights = [weights[y0], weights[~y0]]

        def _loss_for_optimize(params):
            return LogisticRegression._loss(x0, x1, params[0], params[1:], self.lam, loss_weights)
        def _gradient_for_optimize(params):
            return LogisticRegression._gradient_loss(x, y, params[0], params[1:], self.lam, weights)

        params_opt = fmin_bfgs(_loss_for_optimize, np.zeros(1 + x.shape[1]), fprime=_gradient_for_optimize, maxiter=200)

        self.b = params_opt[0]
        self.w = params_opt[1:]

        return _loss_for_optimize(params_opt)

    def save_model(self, model_file):
        """Serialize model to model_file"""
        m = {'b':self.b,
            'w':self.w.tolist()}

        with open(model_file, 'w') as f:
            json.dump(m, f)

    @classmethod
    def load_model(cls, model_file):
        '''If a string is provided, it's assumed to be a path to a file
        containing a JSON blob describing the model. Otherwise, it should
        be a dictionary representing the model'''
        if isinstance(model_file, basestring):
            params = json.load(open(model_file, 'r'))
        else:
            params = model_file
        ret = cls()
        ret.b = float(params['b'])
        ret.w = np.array(params['w'])
        return ret

    @staticmethod
    def _sigmoid(x, b, w):
        """Return sigma(x) = 1.0 / (1.0 + exp(-x * w - b))
        X = N x (nvars)
        
        Computes sigma(w * x) for each data point
        
        Returns a (N, ) array"""
        return np.minimum(np.maximum(1.0 / (1.0 + np.exp(-b - np.sum(w * x, axis=1))), 1.0e-12), 1 - 1.0e-12)

    @staticmethod
    def _loss(x0, x1, b, w, lam, weights=None):
        """Return loss function at x.
        x0 = (N0, nvars) numpy array of x where y == 0
        x1 = (N1, nvars) numpy array of x where y == 1

        loss = Logistic loss + 0.5 * lambda * sum(w**2)
        logistic loss =  -sum ( log(sigmoid(x))   y == 1
                                log(1 - sigmoid(x)) if y == 0 )
        weights = if provided an [(N0, ), (N1, )] list of arrays to add in to each
            observation's contribution to error.
            first entry corresponds to x0, second to x1
        """
        loss = 0.5 * lam * np.sum(w ** 2)
        if weights is None:
            loss += -np.sum(np.log(LogisticRegression._sigmoid(x1, b, w))) - np.sum(np.log(1.0 - LogisticRegression._sigmoid(x0, b, w)))
        else:
            loss += -np.sum(weights[1] * np.log(LogisticRegression._sigmoid(x1, b, w))) - np.sum(weights[0] * np.log(1.0 - LogisticRegression._sigmoid(x0, b, w)))
        return loss

    @staticmethod
    def _gradient_loss(x, y, b, w, lam, weights=None):
        """Return the gradient of the loss.

           x0 = (N, nvars) numpy array of x
           y = 0, 1 prediction

           gradient = logistic gradient loss + self.lam * w
                logistic loss gradient = sum_data (sigmoid(x) - y) * x

            weights = if provided an (N, 2) array to add in to each
                observation's contribution to error.
                first entry corresponds to x0, second to x1
        """
        nvars = len(w)
        gradient = np.zeros(nvars + 1)               # first position is b
        gradient[1:] = lam * w

        # need sum(sigmoid(x) - y) * x for all variables
        error = LogisticRegression._sigmoid(x, b, w) - y
        if weights is None:
            gradient[0] = np.sum(error)   # * 1 for bias term
            for k in xrange(nvars):
                gradient[k + 1] += np.sum(error * x[:, k])
        else:
            gradient[0] = np.sum(error * weights)   # * 1 for bias term
            for k in xrange(nvars):
                gradient[k + 1] += np.sum(weights * error * x[:, k])

        return gradient


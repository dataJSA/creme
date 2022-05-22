import collections
import numbers
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy import stats

from river import base, utils


@dataclass
class GaussianBelief:
    """Gaussian Belief over model weight.

    Represents a gaussian belief over a weight i.e. weight follow normal distribution with parameters μ ``mean`` and σ2 ``variance``.

    Parameters
    ----------
    beta
        Scale of the inverse link function (standard normal cumulative distribution function).
    n_features
        Number of features.
    mean
        Mean of the gaussian belief over the weight.
    prior_probability
        Prior probability on the feature weights.

    Examples
    --------

    >>> from river import datasets
    >>> from river import linear_model

    >>> X_y = datasets.Phishing()

    >>> prior = linear_model.ad_predictor.GaussianBelief(beta=0.05, n_features=X_y.n_features, prior_probability=0.6)

    >>> print(prior)
    Biased normal prior on a weight 𝒩(μ=2.281, σ2=1.000)

    """

    beta: float
    n_features: int
    variance: float = 1.0
    mean: float = field(init=False)
    prior_probability: Union[float, None] = None

    def __post_init__(self):
        self.mean = (
            0
            if self.prior_probability is None
            else stats.norm.ppf(self.prior_probability)
            * (self.beta**2 + self.n_features)
        )

    def __repr__(self) -> str:
        return (
            f"Unbiased normal prior on weights 𝒩(μ={self.mean:.3f}, σ2={self.variance:.3f})"
            if self.prior_probability is None
            else f"Biased normal prior on weights 𝒩(μ={self.mean:.3f}, σ2={self.variance:.3f})"
        )


class AdPredictor(base.Classifier):
    """Online Bayesian Probit Regression algorithm.

    AdPredictor is a bayesian online probit regression model designed for click through rate (CTR) prediction.

    Parameters
    ----------
    beta
        Scale of the inverse link function (standard normal cumulative distribution function).
    n_features
        Number of features.
    prior_probability
        Prior probability on the feature weights.

    Attributes
    ----------
    variance : float
                Variance of the gaussian belief over the weight.

    Notes
    -----
    In (1) the application is focused on discrete multi-valued features where the model is fitted on 1-in-N encoding of the features.
    Using preprocessing is advised.

    References
    ----------
    [^1]: [Graepel, Thore, et al. "Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine." Omnipress, 2010.](https://discovery.ucl.ac.uk/id/eprint/1395202/1/Graepel_901%255B1%255D.pdf)
    [^2]: [He, Xinran, et al. "Practical lessons from predicting clicks on ads at facebook." Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. 2014.](https://dl.acm.org/doi/pdf/10.1145/2648584.2648589)
    [^3]: [Tulloch, Andrew. "Online Learning with Microsoft's AdPredictor algorithm"](http://tullo.ch/articles/online-learning-with-adpredictor/)

    """

    def __init__(self, prior: GaussianBelief, surprise: float, epsilon: float):
        self.prior = prior
        self.surprise = surprise
        self.epsilon = epsilon
        self.beta = prior.beta
        self.prior_probability = prior.prior_probability
        self.weights = collections.defaultdict(prior)

    def _total_mean_variance(self, x):
        """Total mean and variance of the gaussian beliefs over the actives weights for a given instance.

        Parameters
        ----------
        x : dict
           1-in-N encoded features of the instance.

        Returns
        -------
        tuple of float
            Total mean and variance of the gaussian beliefs over the actives weights.
        """
        means = [self.weights[i].mean * xi for i, xi in x.items()]
        variances = [self.weights[i].variance * xi for i, xi in x.items()]
        return sum(means), sum(variances) + self.beta**2

    def _step_size(self, t):
        """Learning step size functions.

        Parameters
        ----------
        t : float

        Returns
        -------
        tuple of float
            Evaluation of the step size functions at the clipped total mean to total standard deviation ratio.
        """
        t = utils.math.clamp(t, minimum=-self.surprise, maximum=self.surprise)
        v = stats.norm.pdf(t) / stats.norm.cdf(t)
        w = v * (v + t)
        return (v, w)

    def _apply_dynamic_corrections(self, weight):
        """Dynamic corrections of the mean and variance of the gaussian belief over a weight.

        Update gaussian belief over a weight to converge back to the prior in the limit of zero data and infinite time.

        Parameters
        ----------
        weight
            Gaussian belief over the weight.

        Returns
        -------
        GaussianBelief
            Mean and variance adjusted gaussian belief.
        """
        posterior = GaussianBelief(
            beta=self.prior.beta, n_features=self.prior.n_features
        )

        adjusted_variance = (
            weight.variance
            * self.prior.variance
            / (
                (1.0 - self.epsilon) * self.prior.variance
                + self.epsilon * weight.variance
            )
        )
        adjusted_mean = adjusted_variance * (
            (1.0 - self.epsilon) * weight.mean / weight.variance
            + self.epsilon * self.prior.mean / self.prior.variance
        )

        posterior.mean, posterior.variance = adjusted_mean, adjusted_variance

        return posterior

    @staticmethod
    def _target_encoding(y):
        assert isinstance(y, numbers.Number)
        return 1.0 if y == 1 else -1.0

    def learn_one(self, x, y):

        y = self._target_encoding(y)

        total_mean, total_variance = self._total_mean_variance(x)
        v, w = self._step_size(y * total_mean / total_variance)

        for i, xi in x.items():

            weight = self.weights[i]

            weight.mean += y * weight.variance / np.sqrt(total_variance) * v
            weight.variance *= 1.0 - weight.variance / total_variance * w

            adjusted_weight = self._apply_dynamic_corrections(weight)

            self.weight[i] = adjusted_weight

        return self

    def predict_proba_one(self, x):

        total_mean, total_variance = self._total_mean_variance(x)
        p = stats.norm.cdf(total_mean / total_variance)

        return {False: 1.0 - p, True: p}
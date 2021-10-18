import collections
import numbers
from dataclasses import dataclass, field
from typing import Union

from river import base, utils
from scipy import stats


@dataclass
class NormalPrior:
    """Normal Prior over model weight.

    The Normal Prior represents a gaussian belief over a weight i.e. weight follow normal distribution with parameters μ ``mean`` and σ2 ``variance``.

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

    >>> prior = linear_model.ad_predictor.NormalPrior(beta=0.05, n_features=X_y.n_features, prior_probability=0.6)

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
            * (self.beta ** 2 + self.n_features)
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

    def __init__(self, prior: NormalPrior, surprise: float, epsilon: float):
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
        return sum(means), sum(variances) + self.beta ** 2

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

    @staticmethod
    def _target_encoding(y):
        assert isinstance(y, numbers.Number)
        return 1.0 if y == 1 else -1.0

    def learn_one(self, x, y):
        pass

    def predict_proba_one(self, x):
        pass

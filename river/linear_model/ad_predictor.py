from typing import Union, NamedTuple
from scipy import stats


class NormalPrior(NamedTuple):
    """Normal Prior over model weight.

    The Normal Prior represents a gaussian belief over a weight i.e. weight follow normal distribution with parameters μ ``mean`` and σ2 ``variance``.

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
    prior_probability: Union[float, None] = None

    @property
    def mean(self):
        """
        mean : float
            Mean of the gaussian belief over the weight.
        """
        return (
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

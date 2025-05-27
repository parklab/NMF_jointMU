"""
Implementation of non-negative matrix factorization (NMF) with
the generalized Kullback-Leibler (KL) divergence.
Both the standard multiplicative updates from Lee & Seung and the
joint update rules are implemented.
"""

from typing import Literal
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from numba import njit

EPSILON = np.finfo(np.float32).eps


@njit
def kl_divergence(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    r"""
    The generalized Kullback-Leibler divergence
    D_KL(X || WH) = \sum_vd X_vd * ln(X_vd / (WH)_vd) - \sum_vd X_vd + \sum_vd (WH)_vd.

    Parameters
    ----------
    X : np.ndarray of shape (n_words, n_samples)
        data matrix

    W : np.ndarray of shape (n_words, n_topics)
        topic matrix

    H : np.ndarray of shape (n_topics, n_samples)
        exposure matrix

    Returns
    -------
    result : float
    """
    V, D = X.shape
    WH = W @ H
    result = 0.0

    for d in range(D):
        for v in range(V):
            if X[v, d] != 0:
                result += X[v, d] * np.log(X[v, d] / WH[v, d])
                result -= X[v, d]
            result += WH[v, d]

    return result


@njit
def update_W(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    update_method: Literal["mu-standard1999", "mu-standard2000"] = "mu-standard2000",
    clip: bool = True,
) -> np.ndarray:
    """
    The standard multiplicative update rule of the
    topic matrix W.
    Clipping the matrix avoids floating point errors.

    Parameters
    ----------
    X : np.ndarray of shape (n_words, n_samples)
        data matrix

    W : np.ndarray of shape (n_words, n_topics)
        topic matrix

    H : np.ndarray of shape (n_topics, n_samples)
        topic weight matrix

    update_method : Literal["mu-standard1999", "mu-standard2000"]
        both the alternating update method from the Lee & Seung nature paper (1999)
        and the Lee & Seung NIPS paper (2000) are supported

    clip : bool, default=True
        If True, clip the matrix to prevent floating-point errors.

    Returns
    -------
    W : np.ndarray
        the updated topic matrix

    References
    ----------
    D. Lee, H. Seung: Learning the parts of objects by non-negative matrix factorization, 1999
    https://www.nature.com/articles/44565

    D. Lee, H. Seung: Algorithms for Non-negative Matrix Factorization
    - Advances in neural information processing systems, 2000
    https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
    """
    aux = X / (W @ H)
    W *= aux @ H.T
    if update_method == "mu-standard2000":
        W /= np.sum(H, axis=1)
    else:  # update_method == "mu-standard1999"
        W /= np.sum(W, axis=0)
    if clip:
        W = W.clip(EPSILON)
    return W


@njit
def update_H(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    update_method: Literal["mu-standard1999", "mu-standard2000"] = "mu-standard2000",
    clip: bool = True,
) -> np.ndarray:
    """
    The standard multiplicative update rule of the
    topic weight matrix H.
    Clipping the matrix avoids floating point errors.

    Parameters
    ----------
    X : np.ndarray of shape (n_words, n_samples)
        data matrix

    W : np.ndarray of shape (n_words, n_topics)
        topic matrix

    H : np.ndarray of shape (n_topics, n_samples)
        topic weight matrix

    update_method : Literal["mu-standard1999", "mu-standard2000"]
        both the alternating update method from the Lee & Seung nature paper (1999)
        and the Lee & Seung NIPS paper (2000) are supported

    clip : bool, default=True
        If True, clip the matrix to prevent floating-point errors.

    Returns
    -------
    H : np.ndarray
        the updated topic weight matrix

    References
    ----------
    D. Lee, H. Seung: Learning the parts of objects by non-negative matrix factorization, 1999
    https://www.nature.com/articles/44565

    D. Lee, H. Seung: Algorithms for Non-negative Matrix Factorization
    - Advances in neural information processing systems, 2000
    https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
    """
    aux = X / (W @ H)
    H *= W.T @ aux
    if update_method == "mu-standard2000":
        H /= np.sum(W, axis=0)[:, np.newaxis]
    if clip:
        H = H.clip(EPSILON)
    return H


@njit
def update_WH(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    update_method: Literal["mu-jointnorm1", "mu-jointnorm2"] = "mu-jointnorm1",
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The joint update rule for the topic matrix W and
    the topic weight matrix H under the constraint of normalized
    topics.
    Clipping the matrix avoids floating point errors.

    Parameters
    ----------
    X : np.ndarray of shape (n_words, n_samples)
        data matrix

    W : np.ndarray of shape (n_words, n_topics)
        topic matrix

    H : np.ndarray of shape (n_topics, n_samples)
        topic weight matrix

    update_method : Literal["mu-jointnorm1", "mu-jointnorm2"]
        "mu-jointnorm1":
            the columns of the topic matrix sum to 1
        "mu-jointnorm2":
            the columns of the topic and topic weight matrix sum to 1 (PLSA)

    clip : bool, default=True
        If True, clip the matrices to prevent floating-point errors.

    Returns
    -------
    W_updated : np.ndarray
        the updated topic matrix

    H : np.ndarray
        the updated topic weight matrix
    """
    aux = X / (W @ H)
    W_updated = W * (aux @ H.T)
    W_updated /= np.sum(W_updated, axis=0)
    H *= W.T @ aux
    if update_method == "mu-jointnorm2":
        H /= np.sum(H, axis=0)
    if clip:
        W_updated, H = W_updated.clip(EPSILON), H.clip(EPSILON)
    return W_updated, H


class KLNMF:
    """
    Decompose a count matrix X into the product of a topic
    matrix W and a topic weight matrix H by minimizing the
    generalized Kullback-Leibler (KL) divergence between
    X and the reconstruction WH.

    References
    ----------
    D. Lee, H. Seung: Learning the parts of objects by non-negative matrix factorization, 1999
    https://www.nature.com/articles/44565

    D. Lee, H. Seung: Algorithms for Non-negative Matrix Factorization
    - Advances in neural information processing systems, 2000
    https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
    """

    def __init__(
        self,
        n_topics: int = 1,
        update_method: Literal[
            "mu-standard1999", "mu-standard2000", "mu-jointnorm1", "mu-jointnorm2"
        ] = "mu-standard2000",
        clip: bool = True,
        min_iterations: int = 10,
        max_iterations: int = 100000,
        conv_test_freq: int = 10,
        tol: float = 1e-5,
    ):
        """
        Parameters
        ----------
        n_topics : int
            number of topics

        update_method : Literal["mu-standard1999", "mu-standard2000", "mu-jointnorm1", "mu-jointnorm2], default=mu-standard2000
            The standard multiplicative update rules alternate between
            optimizing the topics and topic weights.
            The joint multiplicative update rules update both matrices at once.
            They require one matrix multiplication less per iteration.

        clip : bool, default=True
            If True, clip the matrices after each iteration
            to prevent floating-point errors.

        min_iterations : int, default=10
            minimum number of iterations

        max_iterations : int, default=10000
            maximum number of iterations

        conv_test_freq : int
            The frequency at which the algorithm is tested for convergence.
            The objective function value is only computed every 'conv_test_freq'
            many iterations.

        tol : float, default=1e-5
            Tolerance of the stopping condition.
        """
        if update_method not in [
            "mu-standard1999",
            "mu-standard2000",
            "mu-jointnorm1",
            "mu-jointnorm2",
        ]:
            raise ValueError("The provided update method is not supported.")
        self.n_topics = n_topics
        self.update_method = update_method
        self.clip = clip
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.conv_test_freq = conv_test_freq
        self.tol = tol
        self.X = np.empty((0, 0), dtype=float)
        self.W = np.empty((0, 0), dtype=float)
        self.H = np.empty((0, 0), dtype=float)
        self.history: dict[int, tuple[float, float]] = {}

    def initialize(self, X: np.ndarray, seed: int | None = None) -> None:
        """
        Random uniform initialization of the topics W and
        the topic weights H.
        The topics weights are scaled according to the total
        word counts of the documents.
        """
        if seed is not None:
            np.random.seed(seed)

        self.X = X
        n_words, n_samples = X.shape
        self.W = np.random.dirichlet(np.ones(n_words), size=self.n_topics).T
        self.H = np.random.dirichlet(np.ones(self.n_topics), size=n_samples).T
        if self.update_method != "mu-jointnorm2":
            scaling = np.sum(X, axis=0)
            self.H *= scaling

    def objective_function(self) -> float:
        return kl_divergence(self.X, self.W, self.H)

    def update_parameters(self) -> None:
        if self.update_method in ["mu-standard1999", "mu-standard2000"]:
            self.W = update_W(self.X, self.W, self.H, self.update_method, self.clip)
            self.H = update_H(self.X, self.W, self.H, self.update_method, self.clip)
        else:
            self.W, self.H = update_WH(
                self.X, self.W, self.H, self.update_method, self.clip
            )

    def fit(
        self,
        X: np.ndarray,
        seed: int | None = None,
        verbose: int = 0,
    ) -> None:
        """
        Factorize a non-negative count matrix into topic and topic weight matrices.

        Parameters
        ----------
        X : np.ndarray
            A non-negative count matrix of shape (n_words, n_samples), where each column
            represents a sample and each row corresponds to a word or feature.

        seed : int or None, default=None
            Random seed for reproducible initialization. If None, the initialization is random.

        verbose : int, default=0
            Verbosity level. Set to 0 for no output.

        Returns
        -------
        W : np.ndarray
            Topic matrix of shape (n_words, n_topics).

        H : np.ndarray
            Topic weight matrix of shape (n_topics, n_samples).
        """

        self.initialize(X, seed)
        n_iteration = 0
        of_value = self.objective_function()
        of_value_old = of_value
        start = time.time()
        history = {0: (of_value, 0.0)}
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % 1000 == 0:
                print(f"iteration: {n_iteration}; objective: {of_value_old:.2f}")

            self.update_parameters()

            if n_iteration % self.conv_test_freq == 0:
                of_value_old = of_value
                of_value = self.objective_function()
                rel_change = (of_value_old - of_value) / of_value_old
                converged = rel_change < self.tol and n_iteration >= self.min_iterations
                history[n_iteration] = (of_value, time.time() - start)

            if n_iteration == self.max_iterations:
                converged = True
                history[n_iteration] = (self.objective_function(), time.time() - start)

        self.history = history

    def plot_history_iterations(
        self,
        min_iteration: int | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:
        """
        Plot the objective function values for all iterations with
        n_iteration >= 'min_iteration'.
        """
        ns_iteration = np.array(list(self.history.keys()))
        of_values = np.array([v[0] for v in self.history.values()])

        if min_iteration is None:
            min_iteration = self.conv_test_freq
        if min_iteration > ns_iteration[-1]:
            raise ValueError(
                "The smallest iteration number shown in the history plot "
                "cannot be larger than the total number of iterations."
            )

        min_index = next(
            idx
            for idx, n_iteration in enumerate(ns_iteration)
            if n_iteration >= min_iteration
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        ax.set(xlabel="n_iteration", ylabel="objective function value")
        ax.plot(ns_iteration[min_index:], of_values[min_index:], **kwargs)
        return ax

    def plot_history_time(
        self,
        min_time: float | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:
        """
        Plot the objective function values for all times >= 'min_time'.
        """
        of_values = np.array([v[0] for v in self.history.values()])
        times = np.array([v[1] for v in self.history.values()])

        if min_time is None:
            min_time = times[1]
        if min_time > times[-1]:
            raise ValueError(
                "The smallest time shown in the history plot "
                "cannot be larger than the total rumtime."
            )

        min_index = next(idx for idx, time in enumerate(times) if time >= min_time)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.set(xlabel="time (seconds)", ylabel="objective function value")
        ax.plot(times[min_index:], of_values[min_index:], **kwargs)
        return ax

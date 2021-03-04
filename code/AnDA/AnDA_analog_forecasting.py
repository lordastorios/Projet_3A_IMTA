#!/usr/bin/env python

""" AnDA_analog_forecasting.py: Apply the analog method on catalog of historical data to generate forecasts. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np
from AnDA.AnDA_tools import mk_stochastic


def AnDA_analog_forecasting(x, AF):
    """ Apply the analog method on catalog of historical data to generate forecasts. """

    # initializations
    N, n = x.shape
    xf = np.zeros([N, n])
    xf_mean = np.zeros([N, n])
    i_var_neighboor = np.arange(0, n)
    i_var = np.arange(0, n)

    # find the indices and distances of the k-nearest neighbors (knn)
    dist_knn, index_knn = AF.search_neighbors(x)

    # normalisation parameter for the kernels
    lambdaa = np.median(dist_knn)

    # compute weights
    if AF.k == 1:
        weights = np.ones([N, 1])
    else:
        weights = mk_stochastic(np.exp(-np.power(dist_knn, 2) / lambdaa))

    # for each member/particle
    for i_N in range(0, N):

        xf_tmp = np.zeros([AF.k, np.max(i_var) + 1])

        # select the regression method
        if AF.regression == "locally_constant":
            xf_tmp[:, i_var] = AF.catalog.successors[
                np.ix_(index_knn[i_N, :], i_var)
            ]
            # weighted mean and covariance
            xf_mean[i_N, i_var] = np.sum(
                xf_tmp[:, i_var]
                * np.repeat(weights[i_N, :][np.newaxis].T, len(i_var), 1),
                0,
            )
            E_xf = (
                xf_tmp[:, i_var]
                - np.repeat(xf_mean[i_N, i_var][np.newaxis], AF.k, 0)
            ).T
            cov_xf = (
                1.0
                / (1.0 - np.sum(np.power(weights[i_N, :], 2)))
                * np.dot(
                    np.repeat(weights[i_N, :][np.newaxis], len(i_var), 0)
                    * E_xf,
                    E_xf.T,
                )
            )

        elif AF.regression == "increment":
            xf_tmp[:, i_var] = (
                np.repeat(x[i_N, i_var][np.newaxis], AF.k, 0)
                + AF.catalog.successors[np.ix_(index_knn[i_N, :], i_var)]
                - AF.catalog.analogs[np.ix_(index_knn[i_N, :], i_var)]
            )
            # weighted mean and covariance
            xf_mean[i_N, i_var] = np.sum(
                xf_tmp[:, i_var]
                * np.repeat(weights[i_N, :][np.newaxis].T, len(i_var), 1),
                0,
            )
            E_xf = (
                xf_tmp[:, i_var]
                - np.repeat(xf_mean[i_N, i_var][np.newaxis], AF.k, 0)
            ).T
            cov_xf = (
                1.0
                / (1 - np.sum(np.power(weights[i_N, :], 2)))
                * np.dot(
                    np.repeat(weights[i_N, :][np.newaxis], len(i_var), 0)
                    * E_xf,
                    E_xf.T,
                )
            )

        elif AF.regression == "local_linear":
            # NEW VERSION (USING PCA)
            # pca with weighted observations
            mean_x = np.sum(
                AF.catalog.analogs[np.ix_(index_knn[i_N, :], i_var_neighboor)]
                * np.repeat(
                    weights[i_N, :][np.newaxis].T, len(i_var_neighboor), 1
                ),
                0,
            )
            analog_centered = AF.catalog.analogs[
                np.ix_(index_knn[i_N, :], i_var_neighboor)
            ] - np.repeat(mean_x[np.newaxis], AF.k, 0)
            analog_centered = analog_centered * np.repeat(
                np.sqrt(weights[i_N, :])[np.newaxis].T, len(i_var_neighboor), 1
            )
            U, S, V = np.linalg.svd(analog_centered, full_matrices=False)
            coeff = V.T[:, 0:5]

            W = np.sqrt(np.diag(weights[i_N, :]))
            A = np.insert(
                np.dot(
                    AF.catalog.analogs[
                        np.ix_(index_knn[i_N, :], i_var_neighboor)
                    ],
                    coeff,
                ),
                0,
                1,
                1,
            )
            Aw = np.dot(W, A)
            B = AF.catalog.successors[np.ix_(index_knn[i_N, :], i_var)]
            Bw = np.dot(W, B)
            mu = np.dot(
                np.insert(np.dot(x[i_N, i_var_neighboor], coeff), 0, 1),
                np.linalg.lstsq(Aw, Bw)[0],
            )
            pred = np.dot(A, np.linalg.lstsq(A, B)[0])
            res = B - pred
            xf_tmp[:, i_var] = np.tile(mu, (AF.k, 1)) + res
            # weighted mean and covariance
            xf_mean[i_N, i_var] = mu
            if len(i_var) > 1:
                cov_xf = np.cov(res.T)
            else:
                cov_xf = np.cov(res.T)[np.newaxis][np.newaxis]
            # constant weights for local linear
            weights[i_N, :] = 1.0 / len(weights[i_N, :])
        else:
            print(
                "Error: choose AF.regression between 'locally_constant', 'increment', 'local_linear' "
            )
            quit()

        # random sampling from the multivariate Gaussian distribution
        xf[i_N, i_var] = np.random.multivariate_normal(
            xf_mean[i_N, i_var], cov_xf
        )

    return xf, xf_mean
    # end

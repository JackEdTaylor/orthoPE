import orthope
import datahandlers
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

language = 'german'

# get input words from Gagl et al. (2020)
dh = datahandlers.Gagl2020DataHandler(language=language)
input_words = dh.get_unique_words()
n_letters = dh.get_nletter_lims()

# courier new model
est1 = orthope.OrthopeEstimator(language, 'courier', gauss_noise_sd=0.1, n_letters=n_letters, input_words=input_words, data_label='test')

# demonstrate the "drift" noise
_ = est1.__render_text__('Tisch', drift_noise_prop=0.25, max_drift_dist_prop=0.05, show=True)
_ = est1.__render_text__('Tisch', drift_noise_prop=0.25, max_drift_dist_prop=0.50, show=True)
_ = est1.__render_text__('Tisch', drift_noise_prop=0.50, max_drift_dist_prop=0.05, show=True)
_ = est1.__render_text__('Tisch', drift_noise_prop=0.50, max_drift_dist_prop=0.50, show=True)


_ = est1.__render_text__('Tisch', gauss_noise_sd=0.25, show=True)

est1.estimate_corpus_stats()
est1.plot_stat('mu')     # simple average
est1.plot_stat('sigma')  # covariance
est1.plot_stat('kal')    # kalman gain
est1.plot_stat('pi')     # precision matrix
est1.plot_stat('pi_id')  # diagonal of precision matrix

# example opes df
est1_opes_df = est1.__create_opes_df__(words=['Hacee', 'Sanee', 'Häuse', 'Tisch', 'XXXXX'], save=False)

# est1.plot_stat('pi')     # precision matrix (does not converge when gauss_noise_sd==0.0)
# est1.plot_stat('pi_id')  # diagonal of precision matrix (does not converge when gauss_noise_sd==0.0)

# force comic sans to be monospaced
est1_mono = orthope.OrthopeEstimator(language, 'comic', gauss_noise_sd=0.1, n_letters=n_letters, input_words=input_words, data_label='test', force_monospace=True)

# show forced monospace with drift
_ = est1_mono.__render_text__('Tisch', drift_noise_prop=0.5, max_drift_dist_prop=0.1, show=True)
_ = est1_mono.__render_text__('Tisch', drift_noise_prop=1.0, max_drift_dist_prop=0.1, show=True)  # note that the negative image of the word is visible
_ = est1_mono.__render_text__('Tisch', drift_noise_prop=1.0, max_drift_dist_prop=np.inf, show=True)  # show that this problem doesn't apply when max_drift_dist_prop is infinite

est1_mono.estimate_corpus_stats()

est1_mono.plot_stat('mu')     # simple average
est1_mono.plot_stat('sigma')  # covariance
est1_mono.plot_stat('kal')    # kalman gain

est2 = orthope.OrthopeEstimator(language, 'courier', gauss_noise_sd=0.1, n_letters=n_letters, input_words=input_words, data_label='test')
est2.estimate_corpus_stats()

est2.plot_stat('pi')     # precision matrix
est2.plot_stat('pi_id')  # diagonal of precision matrix

# estimate Kalman-weighted prediction error
est1.__estimate_ope__('Tisch', 'kalmanw_pred_err')

# estimate wasserstein / gromov-wasserstein distance from mu
est1.__estimate_ope__('Tisch', 'pred_err_wd')
est1.__estimate_ope__('Tisch', 'pred_err_gwd')  # takes a lot longer!

# simple average with no frequency-weighting (i.e., type-weighting instead of token-weighting)
est3 = orthope.OrthopeEstimator(language, 'courier', gauss_noise_sd=0.0, n_letters=n_letters, input_words=input_words, freq_weight=False, data_label='test')
est3.estimate_corpus_stats()

# compare mu between weighted and unweighted
est1.plot_stat('mu')
est3.plot_stat('mu')
mu1 = est1.corpus_stats['mu']
mu3 = est3.corpus_stats['mu']

# remove shared zeroes
nz = (mu1>0) & (mu3>0)
mu1 = mu1[nz]
mu3 = mu3[nz]

plt.scatter(mu1, mu3, s=0.1)
plt.hist(mu1 - mu3, bins=100)
sp.stats.pearsonr(mu1, mu3)

# use only the top 5% of words, but no frequency weighting
est4 = orthope.OrthopeEstimator(language, 'courier', gauss_noise_sd=0.0, n_letters=n_letters, freq_perc=(95, 100), input_words=input_words, freq_weight=False, data_label='test')
est4.estimate_corpus_stats()
est4.plot_stat('mu')

# estimator using different prior and input fonts
est5 = orthope.OrthopeEstimator(language, 'courier', prior_font='liberationmono', gauss_noise_sd=0.1, n_letters=n_letters, input_words=input_words, data_label='test')
est5.estimate_corpus_stats()
est5.plot_stat('mu')

# prediction error estimator using letter identity
Lest = orthope.LetterOrthopeEstimator(language, gauss_noise_sd=0.1, n_letters=n_letters, input_words=input_words, data_label='test')
Lest.estimate_corpus_stats()
plt.figure()
plt.plot(Lest.corpus_stats['mu'])

# estimator with per-letter barycentre as orthogrpahic prior
otest = orthope.WithinLetterOptimalTransportOrthopeEstimator(language, 'liberationserif', gauss_noise_sd=0.0, n_letters=n_letters, input_words=input_words, data_label='test')
otest.estimate_corpus_stats()

otest.plot_stat('bcs')

otest.__estimate_ope__('Hacee', 'pred_err_wd')
otest.__estimate_ope__('Sanee', 'pred_err_wd')
otest.__estimate_ope__('Häuse', 'pred_err_wd')
otest.__estimate_ope__('Tisch', 'pred_err_wd')
otest.__estimate_ope__('XXXXX', 'pred_err_wd')

# 6-letter estimator with per-letter barycentre as orthogrpahic prior, with larger font size
otest6 = orthope.WithinLetterOptimalTransportOrthopeEstimator(language, 'courier', gauss_noise_sd=0.0, n_letters=(6, 6), input_words=[], data_label='test', font_size=45)
otest6.estimate_corpus_stats()

otest6.plot_stat('bcs')

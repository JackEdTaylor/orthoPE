import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import glob
import re
import pandas as pd
import scipy
import string
from pathlib import Path
import collections
from itertools import groupby
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

import otfuns

#gauss_noise_sds = [1E-10, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
gauss_noise_sds  = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
min_freq_percs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # 100%, top 90% frequent, top 80% frequent, etc.

special  = 'àâäæçéèêëîïôœùûüÿÀÂÄÆÇÉÈÊËÎÏÔŒÙÛÜŸëïöüĳËÏÖÜĲäöüßÄÖÜẞáéíóúñÁÉÍÓÚÑ'  # special characters to include as alphabetic (in addition to string.ascii_letters)

fontpath = Path('fonts')

font_dict   = {'courier'        : fontpath / 'couriernew.ttf',
			   'courieri'       : fontpath / 'couriernewi.ttf',
			   'cambria'        : fontpath / 'cambria.ttf',
			   'verdana'        : fontpath / 'verdana.ttf',
			   'cambriai'       : fontpath / 'cambriai.ttf',
			   'liberationserif': fontpath / 'liberationserif.ttf',
			   'liberationmono' : fontpath / 'liberationmono.ttf',
			   'comic'          : fontpath / 'comic.ttf'}

def add_drift_noise(text_array, drift_noise_prop, max_drift_dist=2, rng=None):
	# function to apply the "drift noise"
	if rng is None:
		rng = np.random
	max_drift_dist = round(max_drift_dist)
	if drift_noise_prop > 0.0 and max_drift_dist > 0.0:
		render_idx = np.transpose(np.where(~np.isnan(text_array)))
		z_px  = text_array==0.0
		nz_px = np.invert(z_px)
		nz_px_idx = np.where(nz_px)
		
		within_dist_idx = (scipy.spatial.distance.cdist(render_idx, np.transpose(nz_px_idx), metric='Euclidean').min(axis=1) <= max_drift_dist).reshape(text_array.shape)

		swap_px = z_px & within_dist_idx
		sw_px_idx = np.where(swap_px)

		n_drift_px = int(np.round(drift_noise_prop * np.sum(nz_px)))
		samp_sw = rng.choice(np.sum(swap_px), size=n_drift_px, replace=False)
		samp_nz = rng.choice(np.sum(nz_px), size=n_drift_px, replace=False)
		text_array_copy = text_array.copy()
		text_array[sw_px_idx[0][samp_sw], sw_px_idx[1][samp_sw]]  = text_array_copy[nz_px_idx[0][samp_nz], nz_px_idx[1][samp_nz]]
		text_array[nz_px_idx[0][samp_nz], nz_px_idx[1][samp_nz]]  = text_array_copy[sw_px_idx[0][samp_sw], sw_px_idx[1][samp_sw]]
	return text_array

class OrthopeEstimator():

	def __init__(self, language, font, gauss_noise_sd, input_words, font_size=28, prior_font=None, force_monospace=False, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, pad_w_per_char=4, pad_top=2, pad_bottom=2, data_label=None, n_threads=None, verbose=True):
		
		self.n_threads = n_threads
		if n_threads is not None:
			# limit threads for this process, to make parallel-friendly
			n_threads = str(n_threads)
			os.environ["OMP_NUM_THREADS"] = n_threads
			os.environ["OPENBLAS_NUM_THREADS"] = n_threads
			os.environ["MKL_NUM_THREADS"] = n_threads
			os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads
			os.environ["NUMEXPR_NUM_THREADS"] = n_threads

		data_label_lab = '' if data_label is None else f'{data_label} '
		freq_wt_lab = 'freq-weighted' if freq_weight else 'freq-unweighted'

		if verbose:
			print(f'{data_label_lab}{language}, font {font}, prior_font {prior_font}, monospace {force_monospace}, gauss_noise_sd {gauss_noise_sd}, letters {n_letters}, freq% {freq_perc}, {freq_wt_lab}')

		self.alphabet = string.ascii_letters + special + ' '

		self.language         = language
		self.font             = font
		self.prior_font       = font if prior_font==None else prior_font
		self.force_monospace  = force_monospace
		self.gauss_noise_sd   = gauss_noise_sd
		self.freq_weight      = freq_weight
		self.input_words      = input_words
		self.font_size	      = font_size
		self.pad_w_per_char   = pad_w_per_char
		self.pad_top          = pad_top
		self.pad_bottom       = pad_bottom
		self.verbose		  = verbose
		
		# Thread-safe random state for each estimator instance
		self.rng = np.random.RandomState()

		# store subset info (two-unit lists/tuples of >= and <= cutoffs)
		#  - if just one number is given, this will be used as both >= and <= cutoff
		if isinstance(n_letters, collections.abc.Iterable) and len(n_letters)==2:
			self.n_letters = n_letters
		else:
			self.n_letters = (n_letters, n_letters)

		if isinstance(freq_perc, collections.abc.Iterable) and len(freq_perc)==2:
			self.freq_perc = freq_perc
		else:
			self.freq_perc = (freq_perc, freq_perc)

		self.datapath = Path('data_repository')
		self.corppath = self.datapath / Path('corpora')
		self.fontpath = fontpath
		self.savepath = Path('models')

		# lookup dictionary for font paths
		self.font_dict = font_dict

		if self.font != 'word':
			self.imagefont = ImageFont.truetype(self.font_dict[self.font], self.font_size)
			self.imagepriorfont = ImageFont.truetype(self.font_dict[self.prior_font], self.font_size)

		data_label = '' if data_label is None else f'{data_label}_'
		opespath_prefix = f'{data_label}{language}_{font}_{prior_font}_gaussnoisesd-{gauss_noise_sd}_letters-{n_letters[0]}-{n_letters[1]}_freqperc-{freq_perc[0]}-{freq_perc[1]}_freqweight-{freq_weight}_mono-{force_monospace}_opes'.replace('.','p')
		self.opespath = self.savepath / f'{opespath_prefix}.csv'

		if not os.path.exists(self.savepath): os.makedirs(self.savepath)
		if not os.path.exists(self.datapath): os.makedirs(self.datapath)

	def __create_opes_df__(self, words, estimates=None, save=True):

		if estimates is None: 
			estimates = ['n_pixels_l1', 'n_pixels_l2', 
						 'pred_err_l1', 'pred_err_l2', 'pw_pred_err', 
						 'pred_err_wd',
						#  'pred_err_gwd',
						 'mahalanobis', 'kalmanw_pred_err']
			
		# estimates that have binarised prediction implemented
		bin_pred_estimates = ['pred_err_l1', 'pred_err_l2', 'pw_pred_err',
							  'mahalanobis', 'kalmanw_pred_err']
		
		bin_thresholds = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # binary thresholds to test
		
		n_obs   = 100 if self.gauss_noise_sd > 0 else 1
		opes_data = {word: {} for word in words}

		# Pre-render all words without noise to cache them (huge speedup!)
		if self.verbose:
			print('Pre-rendering words without noise...')
			words_to_render = tqdm(words)
		else:
			words_to_render = words
		
		word_renders = {word: self.__render_text__(word, gauss_noise_sd=0.0) for word in words_to_render}

		for est in estimates:
			if self.verbose:
				print(f'Computing estimates for {est}')
				words_iterator = tqdm(words)
			else:
				words_iterator = words
			
			for word in words_iterator:
				# Render with noise n_obs times
				if self.gauss_noise_sd > 0:
					x_renders = [word_renders[word] + self.gauss_noise_sd * self.rng.randn(*word_renders[word].shape) for _ in range(n_obs)]
				else:
					x_renders = [word_renders[word]]
				
				if est in bin_pred_estimates:
					for thr in bin_thresholds:
						opes = [self.__estimate_ope_from_render__(x, est, bin_threshold=thr) for x in x_renders]
						est_lab = est if thr is None else est+'_thr_'+str(thr)
						opes_data[word][est_lab+'_mu']  = np.mean(opes)
						opes_data[word][est_lab+'_std'] = np.std(opes)
				else:
					opes = [self.__estimate_ope_from_render__(x, est) for x in x_renders]
					opes_data[word][est+'_mu']  = np.mean(opes)
					opes_data[word][est+'_std'] = np.std(opes)

		opes_df = pd.DataFrame.from_dict(opes_data, orient='index')

		if save:
			opes_df.to_csv(self.opespath)

		return opes_df
	
	def __get_corpus__(self):
		# Available corpora:
		corpora = {
			'german': {'file':'SUBTLEX-DE.tsv'},
			'english':{'file':'SUBTLEX-US.tsv'},
			'french': {'file':'SUBTLEX-FR.tsv'},
			'dutch':  {'file':'SUBTLEX-NL.tsv'}
		}

		# Reading corpus
		datafile = self.corppath / f'{corpora[self.language]["file"]}'

		df = pd.read_csv(datafile, sep='\t', encoding='utf-8',
				   dtype={'word': str, 'raw_freq': int, 'fpmw': float})
		
		# remove any missing values
		is_missing_words = df.word.isna()
		df = df.loc[~is_missing_words, ]
		if any(is_missing_words):
			if self.verbose:
				print(f'Excluded {is_missing_words.sum()} missing words')

		# remove any non-alphabetic words
		nonalph_regex = f'[^{"|".join(self.alphabet)}]'
		is_nonalph = np.array([bool(re.search(nonalph_regex, w)) for w in df.word])
		df = df.loc[~is_nonalph, ]
		if any(is_nonalph):
			if self.verbose:
				print(f'Excluded {is_nonalph.sum()} non-alphabetic words')
		
		# apply filters
		df = df.loc[[len(w)>=self.n_letters[0] and len(w)<=self.n_letters[1] for w in df.word]]

		# apply percentile filter on frequency
		fpmw_filter = [np.percentile(df.fpmw, self.freq_perc[0]), np.percentile(df.fpmw, self.freq_perc[1])]
		df = df.loc[(df.fpmw>=fpmw_filter[0]) & (df.fpmw<=fpmw_filter[1])]

		self.corpus_df = df

		return None

	def __render_corpora__(self):
		if not hasattr(self, 'corpus_df'):
			self.__get_corpus__()

		# Computing corpus at pixel space assuming identical obs_noise
		dd = np.array([self.__render_text__(word, is_prior=True, gauss_noise_sd=self.gauss_noise_sd) for word in self.corpus_df['word']])
		weights = self.corpus_df['fpmw'].to_numpy()

		return dd, weights

	def estimate_corpus_stats(self, weight_by_freq=None, max_iter=1000):

		if weight_by_freq is None:
			weight_by_freq = self.freq_weight

		if self.verbose:
			print('Rendering corpus and estimating stats...')
		
		fit_done = False
		iter_i = 0
		while not fit_done:
			iter_i += 1
			dd, weights = self.__render_corpora__()

			if not weight_by_freq:
				weights = np.ones(weights.shape)

			# Estimating stats
			# print('Estimating mu and sigma...')
			mu    = np.average(dd, axis=0, weights=weights)
			sigma = np.cov(dd, rowvar=False, aweights=weights)
			
			# Precission matrix: exact and assuming independent distributions
			# print('Estimating precision matrices...')
			try:
				pi = scipy.linalg.pinvh(sigma)
			except np.linalg.LinAlgError as e:
				# print(f'LinAlgError: {e}')
				if iter_i < max_iter:
					continue
				pi = np.zeros(sigma.shape)
				pi[:] = np.nan

			if np.any(np.diag(sigma)==0):
				# avoid division by zero when calculating inverse of sigma
				sigma_diag_noise = np.random.normal(0, 1e-12, size=np.diag(sigma).size)
				pi_id = 1 / (np.diag(sigma) + sigma_diag_noise)
			else:
				pi_id = 1 / (np.diag(sigma))

			# Kalman gain assuming same obs_noise in past and current experiences
			# print('Estimating Kalman gain...')
			obs_sigma = self.gauss_noise_sd * np.identity(sigma.shape[0])

			try:
				kal = sigma @ np.linalg.pinv(sigma + obs_sigma)
			except np.linalg.LinAlgError as e:
				# print(f'LinAlgError: {e}')
				if iter_i < max_iter:
					continue
				kal = np.zeros(sigma.shape)
				kal[:] = np.nan

			if self.verbose:
				if iter_i < max_iter:
					print(f'Fit all corpus stats after {iter_i} attempts at rendering')
				else:
					print(f'Failed to fit all corpus stats after {iter_i} attempts at rendering')

			fit_done = True

		self.corpus_stats = {'mu':    mu, 
							 'sigma': sigma,
							 'pi':    pi,
							 'pi_id': pi_id,
							 'kal':   kal,
							 'pi_is_nan': np.all(np.isnan(pi)),
							 'kal_is_nan': np.all(np.isnan(kal))}

		return None
	
	def __plot_2d_from_flat__(self, x_1d, cmap='binary', log_trans=False, **kwargs):
		if log_trans:
			x_1d = np.log(x_1d)

		x_2d = x_1d.reshape(self.full_array_dims)
		fig, ax = plt.subplots()
		im = ax.imshow(x_2d, interpolation='none', cmap=cmap, **kwargs)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='2.5%', pad=0.1)
		fig.colorbar(im, cax=cax, orientation='vertical')
		return fig, ax
	
	def __plot_2dstat__(self, stat, log_trans=False, cmap='binary'):
		if not hasattr(self, 'corpus_stats') or stat not in self.corpus_stats:
			print(f'{stat} not (yet) estimated via estimate_corpus_stats')
		else:
			if log_trans:
				stat_lab = f'Log {stat}'
			else:
				stat_lab = stat

			fig, ax = self.__plot_2d_from_flat__(self.corpus_stats[stat], cmap=cmap, log_trans=log_trans)
			ax.set_title(stat_lab)

			return fig, ax
		
	def __plot_4dstat__(self, stat, log_trans=False):
		if not hasattr(self, 'corpus_stats') or stat not in self.corpus_stats:
			print(f'{stat} not (yet) estimated via estimate_corpus_stats')
		else:
			stat_4d = self.corpus_stats[stat].reshape((self.full_array_dims[0], self.full_array_dims[1], self.full_array_dims[0], self.full_array_dims[1]))

			stat_lab = f'log {stat}' if log_trans else stat

			fig, axs = plt.subplots(ncols=2, nrows=2)
			gs = axs[0, 0].get_gridspec()

			for ax in axs[:2, 0]:
				ax.remove()

			axbig = fig.add_subplot(gs[:2, 0])
			axbig.set_title(f'Full {stat_lab} Matrix')
			if log_trans:
				im1 = axbig.imshow(np.log(self.corpus_stats[stat]), interpolation='none')
			else:
				im1 = axbig.imshow(self.corpus_stats[stat], interpolation='none')
			divider1 = make_axes_locatable(axbig)
			cax1 = divider1.append_axes('right', size='2.5%', pad=0.1)
			fig.colorbar(im1, cax=cax1, orientation='vertical')

			axs[0, 1].set_title(f'Mean of {stat_lab} (Pixel Space)')
			if log_trans:
				im2 = axs[0, 1].imshow(np.mean(np.ma.masked_invalid(np.log(stat_4d)), axis=(2, 3)), interpolation='none')
			else:
				im2 = axs[0, 1].imshow(stat_4d.mean(axis=(2, 3)), interpolation='none')
			divider2 = make_axes_locatable(axs[0, 1])
			cax2 = divider2.append_axes('right', size='2.5%', pad=0.1)
			fig.colorbar(im2, cax=cax2, orientation='vertical')

			axs[1, 1].set_title(f'SD of {stat_lab} (Pixel Space)')
			if log_trans:
				im3 = axs[1, 1].imshow(np.nanstd(np.ma.masked_invalid(np.log(stat_4d)), axis=(2, 3)), interpolation='none')
			else:
				im3 = axs[1, 1].imshow(stat_4d.std(axis=(2, 3)), interpolation='none')
			divider3 = make_axes_locatable(axs[1, 1])
			cax3 = divider3.append_axes('right', size='2.5%', pad=0.1)
			fig.colorbar(im3, cax=cax3, orientation='vertical')

			fig.tight_layout()

			return fig, ax
		
	def plot_stat(self, stat):
		match stat:
			case 'mu':
				self.__plot_2dstat__(stat, log_trans=False, cmap='binary')
			case 'sigma':
				self.__plot_4dstat__(stat)
			case 'pi':
				if np.all(np.isnan(self.corpus_stats['pi'])):
					raise ValueError('pi is nan (did not converge)')
				self.__plot_4dstat__(stat, log_trans=True)
			case 'pi_id':
				if np.all(np.isinf(self.corpus_stats['pi_id'])):
					raise ValueError('pi_id is inf (did not converge)')
				self.__plot_2dstat__(stat, log_trans=True, cmap='viridis')
			case 'kal':
				self.__plot_4dstat__(stat)

	def __estimate_ope__(self, word, estimate, bin_threshold=None):

		x = self.__render_text__(word, gauss_noise_sd=self.gauss_noise_sd)
		# x_no_noise = self.__render_text__(word, gauss_noise_sd=0.0)

		return self.__estimate_ope_from_render__(x, estimate, bin_threshold=bin_threshold)
	
	def __estimate_ope_from_render__(self, x, estimate, bin_threshold=None):
		"""Compute OPE from pre-rendered text array (avoids re-rendering)."""

		if 'n_pixels_' in estimate or '_wd' in estimate or '_gwd' in estimate:
			if bin_threshold is not None:
				raise ValueError(f'Don\'t know how to apply binary prediction threshold to {estimate}')
		else:
			e = x - self.corpus_stats['mu']
			if bin_threshold is not None:
				e = (e > bin_threshold).astype(e.dtype)

		match estimate:
			case 'n_pixels_l1':
				ope = abs(x).sum()
			case 'n_pixels_l2':
				ope = np.linalg.norm(x)
			case 'pred_err_l1':
				# ope = e.sum()
				ope = abs(e).sum()
			case 'pred_err_l2':
				ope = np.linalg.norm(e)
			case 'pw_pred_err':
				if not e.any():
					ope = np.nan
				else:
					ope = np.linalg.norm(e * self.corpus_stats['pi_id'])
			case 'pred_err_wd':
				if self.font == 'word' or self.gauss_noise_sd!=0.0:
					ope = np.nan
				else:
					ope = otfuns.get_w(
						s = x.reshape(self.full_array_dims),
						t = self.corpus_stats['mu'].reshape(self.full_array_dims))
			case 'pred_err_gwd':
				if self.font == 'word' or self.gauss_noise_sd!=0.0:
					ope = np.nan
				else:
					ope = otfuns.get_gw(
						s = x.reshape(self.full_array_dims),
						t = self.corpus_stats['mu'].reshape(self.full_array_dims))
			case 'mahalanobis':
				if self.corpus_stats['pi_is_nan']:
					ope = np.nan
				else:
					if not e.any():
						ope = np.nan
					else:
						ope = (e @ self.corpus_stats['pi'] @ e.T)**.5
			case 'kalmanw_pred_err':
				if self.corpus_stats['kal_is_nan']:
					ope = np.nan
				else:
					ope = np.linalg.norm(self.corpus_stats['kal'] @ e)

		return ope
	
	def __calculate_canvas_dims__(self, input_words=None, force_monospace=None, pad_w_per_char=None, pad_top=None, pad_bottom=None):
		if not hasattr(self, 'corpus_df'):
			self.__get_corpus__()

		if pad_w_per_char is None:
			pad_w_per_char = self.pad_w_per_char

		if pad_top is None:
			pad_top = self.pad_top
		
		if pad_bottom is None:
			pad_bottom = self.pad_bottom

		if input_words is None:
			input_words = self.input_words

		if force_monospace is None:
			force_monospace = self.force_monospace

		if force_monospace:
			# get max width and height for the input and corpus letters
			input_letters = [l for ls in [list(w) for w in input_words] for l in ls]
			corpus_letters = [l for ls in [list(w) for w in self.corpus_df.word] for l in ls]
			test_text = set([*input_letters, *corpus_letters])

			pad_w = int(np.ceil(pad_w_per_char))
			bbox_pad = np.array([-np.ceil(pad_w/2), -np.ceil(pad_top), np.ceil(pad_w/2), np.ceil(pad_bottom)]).astype(int)

		else:
			# get max width and height for the input and corpus words
			test_text = set([*input_words, *self.corpus_df['word']])

			pad_w = int(max([len(w) for w in test_text]) * pad_w_per_char)
			bbox_pad = np.array([-np.ceil(pad_w/2), -np.ceil(pad_top), np.ceil(pad_w/2), np.ceil(pad_bottom)]).astype(int)

		font_bboxes_f = []
		for font in [self.imagefont, self.imagepriorfont]:
			bboxes = [font.getbbox(w, anchor='ms') for w in test_text]
			xmin, ymin = np.array(bboxes)[:, :2].min(axis=0)
			xmax, ymax = np.array(bboxes)[:, 2:].max(axis=0)

			font_bboxes_f.append( np.array([xmin, ymin, xmax, ymax]) + bbox_pad )
		
		font_bboxes_f = np.array(font_bboxes_f)

		# warn if the dimensions mismatch
		if np.unique(font_bboxes_f, axis=0).shape[0] > 1:
			warnings.warn('Mismatch between dimensions of input font and prior font - will use max extents. Check font alignment!')
			font_bboxes_f = np.array([font_bboxes_f[:, :2].min(axis=0),
									  font_bboxes_f[:, 2:].max(axis=0)]).flatten()
		else:
			font_bboxes_f = font_bboxes_f[0, :]

		# store in self
		self.text_bbox_pad = bbox_pad
		self.text_bbox     = list(font_bboxes_f)
		self.canvas_dims   = [sum([abs(self.text_bbox[i]) for i in [0, 2]]),
					  		  sum([abs(self.text_bbox[i]) for i in [1, 3]])]
		self.array_dims    = (self.canvas_dims[1], self.canvas_dims[0])

		# get dimensions for full image (only differs if force_monospace=True)
		if force_monospace:
			self.full_canvas_dims = [self.canvas_dims[0]*max(self.n_letters), self.canvas_dims[1]]
			self.full_array_dims  = (self.full_canvas_dims[1], self.full_canvas_dims[0])
		else:
			self.full_canvas_dims = self.canvas_dims
			self.full_array_dims  = self.array_dims

		return None

	def __render_text__(self, text, is_prior=False, gauss_noise_sd=0.0, drift_noise_prop=0.0, max_drift_dist_prop=0.05, force_monospace=None, pad_w_per_char=None, pad_top=None, pad_bottom=None, show=False):

		if not hasattr(self, 'canvas_dims'):
			self.__calculate_canvas_dims__(pad_w_per_char=pad_w_per_char, pad_top=pad_top, pad_bottom=pad_bottom, force_monospace=force_monospace)

		if pad_w_per_char is None:
			pad_w_per_char = self.pad_w_per_char

		if pad_top is None:
			pad_top = self.pad_top
		
		if pad_bottom is None:
			pad_bottom = self.pad_bottom

		if force_monospace is None:
			force_monospace = self.force_monospace

		if force_monospace:
			array_2d_list = [ self.__render_text__(text=L, is_prior=is_prior, gauss_noise_sd=gauss_noise_sd, drift_noise_prop=drift_noise_prop, force_monospace=False, pad_w_per_char=pad_w_per_char, pad_top=pad_top, pad_bottom=pad_bottom, show=show).reshape(self.array_dims) for L in text]
			text_array_2d = np.hstack(array_2d_list)
			text_array    = text_array_2d.flatten()
		else:
			# set up font
			if is_prior:
				font = self.imagepriorfont
			else:
				font = self.imagefont

			# Rendering text with pillow
			render   = Image.new('L', self.canvas_dims, color=0)
			draw     = ImageDraw.Draw(render)
			text_pos = (self.canvas_dims[0]/2, -self.text_bbox[1])
			draw.text(text_pos, text, anchor='ms', fill=255, font=font)
			render_array = np.array(render) / 255 # Normalise to r \in [0, 1]

			# Applying "proportional drift" noise
			# (a proportion of non-zero pixels are swapped with pixels that = 0.0 and that are within the max drift distance)
			if drift_noise_prop > 0:
				text_bbox_height = self.array_dims[0] - self.text_bbox_pad[1] - self.text_bbox_pad[3]  # height of the arrays without padding
				max_drift_dist = round(max_drift_dist_prop * text_bbox_height)  # proportion of the text height

				if max_drift_dist < 1.0:
					warnings.warn('max_drift_dist_prop * text height produces a drift distance of <1, so no drift will be applied')
				else:
					render_array = add_drift_noise(render_array, drift_noise_prop=drift_noise_prop, max_drift_dist=max_drift_dist, rng=self.rng)

			# Applying additive Gaussian noise
			noise_array  = gauss_noise_sd * self.rng.randn(*render_array.shape)
			text_array   = (render_array + noise_array).flatten()

			if show:
				plt.figure()
				plt.imshow(render_array + noise_array, cmap='gray', vmin=0.0, vmax=max([text_array.max(), 1.0]), interpolation='none')
				plt.colorbar()

		return text_array
	
	def __get_letter_space_locs_from_xmax__(self, x_2d, expected_spaces=None, show=False):

		# Detects the locations of spaces between letters, assuming that there are no breaks along the x axis within glyphs of a width greater than 12 pixels.
		max_xaxis = x_2d.max(axis=0)

		# dummy code the start and end to the max, so they can be used to detect start and end of word (because of the peak-finding algorithm used)
		max_xaxis[0] = max_xaxis.max()
		max_xaxis[len(max_xaxis)-1] = max_xaxis.max()

		# use peak-finding algorithm to get the spaces' locations
		space_centres, _ = sp.signal.find_peaks(-max_xaxis, distance=4)  # minimum distance is quite low, because of proportional fonts

		# now ignore the zeroes at the starts and ends of the words
		space_locs = space_centres[1:-1]

		# get N deepest troughs
		if expected_spaces is None:
			expected_spaces = self.n_letters[0]-1
		space_locs_idx = np.argpartition(-max_xaxis[space_locs], -expected_spaces)[-expected_spaces:]

		space_locs = space_locs[space_locs_idx]

		if show:
			plt.imshow(x_2d, interpolation='none', cmap='Greys')
			plt.vlines(space_locs, ymin=0, ymax=x_2d.shape[0])
			plt.show()

		space_locs = np.sort(space_locs)

		assert len(space_locs) >= self.n_letters[0]-1, f'Detected {len(space_locs)} spaces in a word image, but expected the min to be {self.n_letters[0]-1}'
		assert len(space_locs) <= self.n_letters[1]-1, f'Detected {len(space_locs)} spaces in a word image, but expected the max to be {self.n_letters[1]-1}'

		return space_locs
	
	def __split_word_img_letters__(self, x_2d, word, is_prior=False, pad_w_per_char=None, pad_top=None, pad_bottom=None, use_cross_cor_meth=False, **kwargs):
		if pad_w_per_char is None:
			pad_w_per_char = self.pad_w_per_char

		if pad_top is None:
			pad_top = self.pad_top
		
		if pad_bottom is None:
			pad_bottom = self.pad_bottom

		# Input x_2d should be a 2d array of the word, with no noise.
		# Returns an image for each detected letter in the word, with zeroes where the other characters were (can preserve the dimensions of the input in each output)

		if use_cross_cor_meth:
			# the result will be approximate, but handles characters that overlap on the x axis

			# get arrays for each character
			char_arrs = [self.__crop_to_content__(self.__render_text__(L, is_prior=is_prior, pad_w_per_char=pad_w_per_char, pad_top=pad_top, pad_bottom=pad_bottom).reshape(self.full_array_dims))[0] for L in word]

			# for each character, get the location in the image

			# first, use 2d cross correlation to find likely locations
			xy_cors = np.array([np.round(sp.signal.correlate(x_2d, y, method='fft', mode='same'), 5) for y in char_arrs])  # rounded for floating point precision in fft

			xmins = []
			x_2d_spl = []

			while len(x_2d_spl)<len(word):
				# assign the letter that matches with the current highest correlation
				max_cors_per_L = xy_cors.max(axis=(1,2))
				L_i = np.argmax(max_cors_per_L)

				max_cor_idx = np.where(xy_cors[L_i, :, :] == max_cors_per_L[L_i])

				# get border of letter
				xmin = int( max_cor_idx[1][0] - np.ceil(char_arrs[L_i].shape[1]/2) )
				# xmax = int( max_cor_idx[1][0] + np.ceil(char_arrs[L_i].shape[1]/2) )
				xmins.append( xmin )

				ymin = int( max_cor_idx[0][0] - np.ceil(char_arrs[L_i].shape[0]/2) )
				# ymax = int( max_cor_idx[0][0] + np.ceil(char_arrs[L_i].shape[0]/2) )

				arr_L_i = np.pad(
					char_arrs[L_i],
					[[ymin, x_2d.shape[0]-char_arrs[L_i].shape[0]-ymin],
					[xmin, x_2d.shape[1]-char_arrs[L_i].shape[1]-xmin]]
				)
				x_2d_spl.append( arr_L_i )

				# zero-out the correlations for this character
				xy_cors[L_i, :, :] = 0.0

			# sort by where the letters start on the x axis
			xmins_as = np.argsort(xmins)
			x_2d_spl = np.array(x_2d_spl)[xmins_as, :, :]

		else:
			# just use the max on the x axis
			space_locs = self.__get_letter_space_locs_from_xmax__(x_2d=x_2d, expected_spaces=len(word)-1, **kwargs)

			space_locs = np.insert(space_locs, 0, 0.0)

			x_2d_spl = []
			for i in range(len(space_locs)):
				x_2d_i = x_2d.copy()

				if i > 0:
					x_2d_i[:, :space_locs[i]] = 0.0

				if i < len(space_locs)-1:
					x_2d_i[:, space_locs[i+1]:] = 0.0
					
				x_2d_spl.append( x_2d_i )

			x_2d_spl = np.array(x_2d_spl)

		return x_2d_spl
	
	def __crop_to_content__(self, x, y_2d=None):
		if len(x.shape)==2:
			# find non-zero elements
			nonzero_idx = np.where(x!=0.0) if y_2d is None else np.where(y_2d!=0.0)
			# create list of slices by which x should be indexed
			crop_idx = [slice(np.min(d), np.max(d)+1) for d in nonzero_idx]
			x_crop = x[tuple(crop_idx)]
		elif len(x.shape)==3:
			# find non-zero elements (averaging across first axis of x)
			nonzero_idx = np.where(x.max(axis=0)!=0.0) if y_2d is None else np.where(y_2d!=0.0)
			# create list of slices by which x should be indexed
			crop_idx = [slice(np.min(d), np.max(d)+1) for d in nonzero_idx]
			crop_idx.insert(0, slice(None))
			x_crop = x[tuple(crop_idx)]
		else:
			raise ValueError('x must be 2d or 3d')
		# calculate the pad_width required to undo the crop
		pad_width = []
		for i, slice_i in enumerate(crop_idx):
			pad_width_i = []
			if slice_i.start is None:
				pad_width_i.append(0)
			else:
				pad_width_i.append(slice_i.start)
			if slice_i.stop is None:
				pad_width_i.append(0)
			else:
				pad_width_i.append(x.shape[i] - slice_i.stop)
			pad_width.append(pad_width_i)

		return(x_crop, pad_width)

	def load_opes(self, input_words=None, save=True, load_existing=True):

		if input_words is None:
			input_words = self.input_words

		if load_existing and os.path.exists(self.opespath):
			if self.verbose:
				print('Loading existing oPE file...')
			opes_df = pd.read_csv(self.opespath)
			# CSV interprets index info as an unnamed column
			opes_df.rename(columns={'Unnamed: 0':'word'}, inplace=True)

			if any([w_i not in set(input_words) for w_i in opes_df.word.unique()]) or any([w_i not in opes_df.word.unique() for w_i in set(input_words)]):
				warnings.warn(f'Loaded oPE file, but mismatch in words!')
		else:
			if self.verbose:
				print(f'Calculating oPE for {len(input_words)} inputs...')
			self.estimate_corpus_stats(weight_by_freq=self.freq_weight)
			opes_df = self.__create_opes_df__(words=input_words, save=save)

		return opes_df


class LetterOrthopeEstimator(OrthopeEstimator):
	
	def __init__(self, language, gauss_noise_sd, input_words, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None, n_threads=None, verbose=True):
		super().__init__(language, font='word', gauss_noise_sd=gauss_noise_sd, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, freq_weight=freq_weight, data_label=data_label, n_threads=n_threads, verbose=verbose)

	def __render_text__(self, text, is_prior=False, pad_w_per_char=None, pad_top=None, pad_bottom=None, gauss_noise_sd=0.0, show=False):
		# note that is_prior and the pad_* arguments are ignored for LetterOrthopeEstimator

		# Settings
		alphabet = self.alphabet

		max_n_letters = max([len(text)]) if np.isinf(self.n_letters[1]) else self.n_letters[1]

		render_array = np.zeros((max_n_letters, len(alphabet)))
		for cix, c in enumerate(text):
			render_array[cix, alphabet.index(c)] = 1
		
		noise_array = gauss_noise_sd * self.rng.randn(*render_array.shape)
		text_array  = (render_array + noise_array).flatten()

		return text_array

	
class WithinLetterOptimalTransportOrthopeEstimator(OrthopeEstimator):
	# this function is more efficient, but assumes earlier on that mass is only transported within letter slots

	def __init__(self, language, font, gauss_noise_sd, input_words, font_size=28, prior_font=None, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, pad_w_per_char=4, pad_top=2, pad_bottom=2, data_label=None, n_threads=None, verbose=True):
		super().__init__(language, font=font, gauss_noise_sd=gauss_noise_sd, input_words=input_words, font_size=font_size, prior_font=prior_font, force_monospace=False, n_letters=n_letters, freq_perc=freq_perc, freq_weight=freq_weight, pad_w_per_char=pad_w_per_char, pad_top=pad_top, pad_bottom=pad_bottom, data_label=data_label, n_threads=n_threads, verbose=verbose)

		# separate opespath if using the optimal transport estimator
		data_label = '' if data_label is None else f'{data_label}_'
		opespath_prefix = f'{data_label}{language}_{font}_{prior_font}_gaussnoisesd-{gauss_noise_sd}_letters-{n_letters[0]}-{n_letters[1]}_freqperc-{freq_perc[0]}-{freq_perc[1]}_freqweight-{freq_weight}_mono-True_opes_wlot'.replace('.','p')  # add "_wlot" suffix
		self.opespath = self.savepath / f'{opespath_prefix}.csv'

		assert self.font != 'word', 'LetterOrthopeEstimator() not implemented for WithinLetterOptimalTransportOrthopeEstimator()'

		if self.gauss_noise_sd != 0.0:
			warnings.warn('Additive Gaussian noise likely to cause problems for the optimal transport estimator')

	def __calculate_canvas_dims__(self, input_words=None, force_monospace=None, pad_w_per_char=None, pad_top=None, pad_bottom=None):
		# for this class, the canvas dimensions are only ever one character in width
		# force_monospace is ignored for this class
		if not hasattr(self, 'corpus_df'):
			self.__get_corpus__()

		if pad_w_per_char is None:
			pad_w_per_char = self.pad_w_per_char

		if pad_top is None:
			pad_top = self.pad_top
		
		if pad_bottom is None:
			pad_bottom = self.pad_bottom

		if input_words is None:
			input_words = self.input_words

		# get max width and height for the input and corpus letters
		input_letters = [l for ls in [list(w) for w in input_words] for l in ls]
		corpus_letters = [l for ls in [list(w) for w in self.corpus_df.word] for l in ls]
		test_letters = set([*input_letters, *corpus_letters])

		pad_w = int(np.ceil(pad_w_per_char))
		bbox_pad = np.array([-np.ceil(pad_w/2), -np.ceil(pad_top), np.ceil(pad_w/2), np.ceil(pad_bottom)]).astype(int)	
		
		font_bboxes_f = []
		for font in [self.imagefont, self.imagepriorfont]:
			bboxes = [font.getbbox(L, anchor='ms') for L in test_letters]
			xmin, ymin = np.array(bboxes)[:, :2].min(axis=0)
			xmax, ymax = np.array(bboxes)[:, 2:].max(axis=0)

			font_bboxes_f.append( np.array([xmin, ymin, xmax, ymax]) + bbox_pad )
		
		font_bboxes_f = np.array(font_bboxes_f)

		# warn if the dimensions mismatch
		if np.unique(font_bboxes_f, axis=0).shape[0] > 1:
			warnings.warn('Mismatch between dimensions of input font and prior font - will not make any attempt to align the fonts, and will use max extents!')
			font_bboxes_f = np.array([font_bboxes_f[:, :2].min(axis=0),
							   		  font_bboxes_f[:, 2:].max(axis=0)]).flatten()
		else:
			font_bboxes_f = font_bboxes_f[0, :]

		# store in self
		self.text_bbox_pad = bbox_pad
		self.text_bbox     = list(font_bboxes_f)
		self.canvas_dims   = [sum([abs(self.text_bbox[i]) for i in [0, 2]]),
					  		  sum([abs(self.text_bbox[i]) for i in [1, 3]])]
		self.array_dims    = (self.canvas_dims[1], self.canvas_dims[0])
		self.full_canvas_dims = self.canvas_dims
		self.full_array_dims  = self.array_dims

		return None

	def __render_corpora__(self):
		if not hasattr(self, 'corpus_df'):
			self.__get_corpus__()

		# for each slot, render all letters that occur in that slot
		words_letts = [list(w) for w in self.corpus_df.word]

		# (currently assumes that all words have the same length)
		slot_letts_unique = [np.unique([wl[i] for wl in words_letts],
								 return_inverse=True, return_counts=True)
								 for i in range(self.n_letters[1])]
		
		slot_letts        = [slu[0] for slu in slot_letts_unique]
		slot_letts_idx    = [slu[1] for slu in slot_letts_unique]  # for each word, the corresponding letter index for each slot
		slot_letts_counts = [slu[2] for slu in slot_letts_unique]

		# Computing corpus at pixel space assuming identical obs_noise
		dd = [np.array(
			[self.__render_text__(sl, is_prior=True, gauss_noise_sd=self.gauss_noise_sd) for sl in slot_letts_i]
			) for slot_letts_i in slot_letts]
		lett_weights = slot_letts_counts

		fpmw = self.corpus_df['fpmw'].to_numpy()

		# weight the letter counts by corresponding word frequencies
		word_weights = [np.array(
			[np.sum(slc[i] * fpmw[sli == i]) for i in range(len(slc))]
			) for slc, sli in zip(slot_letts_counts, slot_letts_idx)]

		return dd, word_weights, lett_weights

	def __create_opes_df__(self, words, estimates=None, save=True):

		if estimates is None: 
			estimates = ['pred_err_l1', 'pred_err_l2', 
						 'pred_err_wd',
						#  'pred_err_gwd',
						]
		
		n_obs   = 100 if self.gauss_noise_sd > 0 else 1
		opes_data = {word: {} for word in words}

		# Pre-render all words and letters without noise to cache them (huge speedup!)
		if self.verbose:
			print('Pre-rendering words without noise...')
			words_to_render = tqdm(words)
		else:
			words_to_render = words
		
		word_renders = {word: [self.__render_text__(L, gauss_noise_sd=0.0) for L in list(word)] for word in words_to_render}

		for est in estimates:
			if self.verbose:
				print(f'Computing estimates for {est}')
				words_iterator = tqdm(words)
			else:
				words_iterator = words
			
			for word in words_iterator:
				# Render with noise n_obs times
				if self.gauss_noise_sd > 0:
					x_renders = [[L_render + self.gauss_noise_sd * self.rng.randn(*L_render.shape) for L_render in word_renders[word]] for _ in range(n_obs)]
				else:
					x_renders = [word_renders[word] for _ in range(n_obs)]
				
				opes = [self.__estimate_ope_from_renders__(x, est) for x in x_renders]
				opes_data[word][est+'_mu']  = np.mean(opes)
				opes_data[word][est+'_std'] = np.std(opes)

		opes_df = pd.DataFrame.from_dict(opes_data, orient='index')

		if save:
			opes_df.to_csv(self.opespath)

		return opes_df
	
	def estimate_corpus_stats(self, weight_by_freq=None):
		if weight_by_freq is None:
			weight_by_freq = self.freq_weight
		
		if self.verbose:
			print('Rendering corpus...')
		dd, word_weights, lett_weights = self.__render_corpora__()
		dd_2d = [[dd_ij.reshape(self.full_array_dims) for dd_ij in dd_i] for dd_i in dd]

		# if weight_by_freq, then the weights will be frequency-weighted...
		if weight_by_freq:
			weights = [w / np.sum(w) for w in word_weights]
		# ...otherwise, use the letter counts (comparable to the other classes)
		else:
			weights = [w / np.sum(w) for w in lett_weights]

		if self.verbose:
			print('Estimating within-letter barycentres...')
		bcs = [otfuns.get_w_barycentre(np.array(L), debias=False, weights=w, reg=0.001, numItermax=int(1e7)) for L, w in zip(dd_2d, weights)]

		# join into a single image
		bcs_joined = np.hstack(bcs)

		self.corpus_stats = {'bcs': bcs, 'bcs_joined': bcs_joined}

		return None
	
	def plot_stat(self, stat):
		if stat=='bcs_joined': stat = 'bcs'

		match stat:
			case 'bcs':
				fig, ax = plt.subplots()
				im = ax.imshow(self.corpus_stats['bcs_joined'], interpolation='none', cmap='binary')
				divider = make_axes_locatable(ax)
				cax = divider.append_axes('right', size='2.5%', pad=0.1)
				fig.colorbar(im, cax=cax, orientation='vertical')
				stat_lab = stat
		
		ax.set_title(stat_lab)
		return fig, ax
	
	def __estimate_ope_from_renders__(self, x_list, estimate):
		"""Compute OPE from pre-rendered letter list (avoids re-rendering)."""
		x_2d = [x_i.reshape(self.full_array_dims) for x_i in x_list]

		e = [x_2d_i - bc_i for x_2d_i, bc_i in zip(x_2d, self.corpus_stats['bcs'])]

		match estimate:
			case 'pred_err_l1':
				ope = abs(np.hstack(e)).sum()
			case 'pred_err_l2':
				ope = np.linalg.norm(np.hstack(e))
			case 'pred_err_wd':
				if self.gauss_noise_sd!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_w(s = L, t = bc_i) for L, bc_i in zip(x_2d, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
			case 'pred_err_gwd':
				if self.gauss_noise_sd!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_gw(s = L, t = bc_i) for L, bc_i in zip(x_2d, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
		return ope
	
	def __estimate_ope__(self, word, estimate):

		x = [self.__render_text__(L, gauss_noise_sd=self.gauss_noise_sd) for L in list(word)]
		x_2d = [x_i.reshape(self.full_array_dims) for x_i in x]

		e = [x_2d_i - bc_i for x_2d_i, bc_i in zip(x_2d, self.corpus_stats['bcs'])]

		match estimate:
			case 'pred_err_l1':
				ope = abs(np.hstack(e)).sum()
			case 'pred_err_l2':
				ope = np.linalg.norm(np.hstack(e))
			case 'pred_err_wd':
				if self.gauss_noise_sd!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_w(s = L, t = bc_i) for L, bc_i in zip(x_2d, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
			case 'pred_err_gwd':
				if self.gauss_noise_sd!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_gw(s = L, t = bc_i) for L, bc_i in zip(x_2d, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
		return ope


def run_all_oPEs(language, input_words, n_letters=(5, 5), data_label=None, n_jobs=1, save_at_each=True, joblib_backend='loky'):
	
	if n_jobs != 1:
		# function that will be called in parallel
		def do_load_opes(gg_i):
			df = gg_i.load_opes(save=save_at_each, load_existing=save_at_each)
			if save_at_each:
				return None
			else:
				return df, gg_i.opespath
		
		tqdm_desc = f'{data_label} {language}, letters {n_letters}'

		# Letter identity approach
		ggs_li = [
			LetterOrthopeEstimator(language=language, gauss_noise_sd=gauss_noise_sd, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight, n_threads=1, verbose=False)
			for gauss_noise_sd in gauss_noise_sds
			for freq_min in min_freq_percs
			for freq_weight in (True, False)
		]

		# Optimal Transport approach
		ggs_ot = [
			WithinLetterOptimalTransportOrthopeEstimator(language=language, font=font, prior_font=None, gauss_noise_sd=0.0, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight, n_threads=1, verbose=False)
			for font in font_dict.keys()
			for freq_min in min_freq_percs
			for freq_weight in (True, False)
		]

		# Euclidean approach
		ggs_euc = [
			OrthopeEstimator(language=language, font=font, prior_font=None, gauss_noise_sd=gauss_noise_sd, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight, force_monospace=force_monospace, n_threads=1, verbose=False)
			for font in font_dict.keys()
			for gauss_noise_sd in gauss_noise_sds
			for freq_min in min_freq_percs
			for freq_weight in (True, False)
			for force_monospace in (True, False)
		]
	
		# Estimate all in parallel
		out = Parallel(n_jobs=n_jobs, backend=joblib_backend, timeout=8**8)(delayed(do_load_opes)(gg_i) for gg_i in tqdm([*ggs_li, *ggs_ot, *ggs_euc], desc=tqdm_desc))

		# save all at end if this is set
		if not save_at_each:
			for out_i in out:
				out_i[0].to_csv(out_i[1])

	else:
		# Letter identity approach
		for gauss_noise_sd in gauss_noise_sds:
			for freq_min in min_freq_percs:
				for freq_weight in (True, False):
					gg = LetterOrthopeEstimator(language=language, gauss_noise_sd=gauss_noise_sd, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight)
					gg.load_opes()

		# Optimal Transport approach
		for font in font_dict.keys():
			for freq_min in min_freq_percs:
				for freq_weight in (True, False):
					gg = WithinLetterOptimalTransportOrthopeEstimator(language=language, font=font, prior_font=None, gauss_noise_sd=0.0, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight)
					gg.load_opes()

		# Euclidean approach
		for font in font_dict.keys():
			for gauss_noise_sd in gauss_noise_sds:
				for freq_min in min_freq_percs:
					for freq_weight in (True, False):
						for force_monospace in (True, False):
							gg = OrthopeEstimator(language=language, font=font, prior_font=None, gauss_noise_sd=gauss_noise_sd, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight, force_monospace=force_monospace)
							gg.load_opes()

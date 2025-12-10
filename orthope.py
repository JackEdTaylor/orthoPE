import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import re
import pandas as pd
import scipy
import string
from pathlib import Path
import collections
from itertools import groupby
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import warnings

import otfuns

#noises = [1E-10, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
noises  = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
min_freq_percs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # 100%, top 90% frequent, top 80% frequent, etc.

special  = 'àâäæçéèêëîïôœùûüÿÀÂÄÆÇÉÈÊËÎÏÔŒÙÛÜŸëïöüĳËÏÖÜĲäöüßÄÖÜẞáéíóúñÁÉÍÓÚÑ'  # special characters to include as alphabetic (in addition to string.ascii_letters)

# self = OrthopeEstimator('german', 'courier', 0.0, ['Tisch', 'Lampe'], data_label='test')
# self = OrthopeEstimator('german', 'verdana', 0.0, ['Tisch', 'Lampe'], data_label='test')
# self = OptimalTransportOrthopeEstimator('german', 'courier', 0.0, ['Tisch', 'Lampe'], data_label='test')
# self = OptimalTransportOrthopeEstimator('german', 'verdana', 0.0, ['Tisch', 'Lampe'], data_label='test')
# self = WithinLetterOptimalTransportOrthopeEstimator('german', 'courier', 0.0, ['Tisch', 'Lampe'], data_label='test')
# self = WithinLetterOptimalTransportOrthopeEstimator('german', 'verdana', 0.0, ['Tisch', 'Lampe'], data_label='test')

class OrthopeEstimator():

	def __init__(self, language, font, noise, input_words, prior_font=None, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None):
		data_label_lab = '' if data_label is None else f'{data_label} '
		freq_wt_lab = 'freq-weighted' if freq_weight else 'freq-unweighted'
		print(f'{data_label_lab}{language}, font {font}, prior_font {prior_font}, noise {noise}, letters {n_letters}, freq% {freq_perc}, {freq_wt_lab}')

		self.alphabet = string.ascii_letters + special + ' '

		self.language    = language
		self.font        = font
		self.prior_font  = font if prior_font==None else prior_font
		self.noise       = noise
		self.freq_weight = freq_weight
		self.input_words = input_words
		self.font_size	 = 34

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
		self.fontpath = Path('fonts')
		self.savepath = Path('models')

		# lookup dictionary for font paths
		self.font_dict   = {'courier'        : self.fontpath / 'couriernew.ttf',
							'courieri'       : self.fontpath / 'couriernewi.ttf',
							'cambria'        : self.fontpath / 'cambria.ttf',
							'verdana'        : self.fontpath / 'verdana.ttf',
							'cambriai'       : self.fontpath / 'cambriai.ttf',
							'liberationserif': self.fontpath / 'liberationserif.ttf',
							'liberationmono' : self.fontpath / 'liberationmono.ttf',
							'comic'          : self.fontpath / 'comic.ttf'}

		data_label = '' if data_label is None else f'{data_label}_'
		opespath_prefix = f'{data_label}{language}_{font}_{prior_font}_noise-{noise}_letters-{n_letters[0]}-{n_letters[1]}_freqperc-{freq_perc[0]}-{freq_perc[1]}_freqweight-{freq_weight}_opes'.replace('.','p')
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
		
		n_obs   = 100 if self.noise > 0 else 1
		opes_df = pd.DataFrame(index=words)

		for est in estimates:
			print(f'Computing estimates for {est}')
			for word in tqdm(words):
				opes = [self.__estimate_ope__(word,est) for _ in range(n_obs)]
				opes_df.at[word, est+'_mu']  = np.mean(opes)
				opes_df.at[word, est+'_std'] = np.std(opes)

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
			print(f'Excluded {is_missing_words.sum()} missing words')

		# remove any non-alphabetic words
		nonalph_regex = f'[^{"|".join(self.alphabet)}]'
		is_nonalph = np.array([bool(re.search(nonalph_regex, w)) for w in df.word])
		df = df.loc[~is_nonalph, ]
		if any(is_nonalph):
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
		dd = np.array([self.__render_text__(word, is_prior=True, noise=self.noise) for word in self.corpus_df['word']])
		weights = self.corpus_df['fpmw'].to_numpy()

		return dd, weights

	def estimate_corpus_stats(self, weight_by_freq=True):
		
		print('Rendering corpus...')
		dd, weights = self.__render_corpora__()

		if not weight_by_freq:
			weights = np.ones(weights.shape)

		# Estimating stats
		print('Estimating mu and sigma...')
		mu    = np.average(dd, axis=0, weights=weights)
		sigma = np.cov(dd, rowvar=False, aweights=weights)
		
		# Precission matrix: exact and assuming independent distributions
		print('Estimating precision matrices...')
		try:
			pi = scipy.linalg.pinvh(sigma)
		except np.linalg.LinAlgError as e:
			print(f'LinAlgError: {e}')
			pi = np.nan

		pi_id = 1 / (np.diag(sigma))

		# Kalman gain assuming same obs_noise in past and current experiences
		print('Estimating Kalman gain...')
		obs_sigma = self.noise * np.identity(sigma.shape[0])

		try:
			kal = sigma @ np.linalg.pinv(sigma + obs_sigma)
		except np.linalg.LinAlgError as e:
			print(f'LinAlgError: {e}')
			kal = np.nan

		self.corpus_stats = {'mu':    mu, 
							 'sigma': sigma,
							 'pi':    pi,
							 'pi_id': pi_id,
							 'kal':   kal}

		return None
	
	def __plot_2d_from_flat__(self, x_1d, cmap='binary', **kwargs):
		x_2d = x_1d.reshape(self.array_dims)
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
				stat_2d = np.log(stat_2d)
				stat_lab = f'Log {stat}'
			else:
				stat_lab = stat

			fig, ax = self.__plot_2d_from_flat__(self.corpus_stats[stat], cmap=cmap)
			ax.set_title(stat_lab)

			return fig, ax
		
	def __plot_4dstat__(self, stat, log_trans=False):
		if not hasattr(self, 'corpus_stats') or stat not in self.corpus_stats:
			print(f'{stat} not (yet) estimated via estimate_corpus_stats')
		else:
			stat_4d = self.corpus_stats[stat].reshape((self.array_dims[0], self.array_dims[1], self.array_dims[0], self.array_dims[1]))

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
				self.__plot_4dstat__(stat, log_trans=True)
			case 'pi_id':
				self.__plot_2dstat__(stat, log_trans=True, cmap='viridis')
			case 'kal':
				self.__plot_4dstat__(stat)

	def __estimate_ope__(self, word, estimate):

		x = self.__render_text__(word, noise=self.noise)
		# x_no_noise = self.__render_text__(word, noise=0.0)

		if 'n_pixels_' not in estimate and '_wd' not in estimate and '_gwd' not in estimate:
			e = x - self.corpus_stats['mu']

		match estimate:
			case 'n_pixels_l1':
				ope = x.sum()
			case 'n_pixels_l2':
				ope = np.linalg.norm(x)
			case 'pred_err_l1':
				# ope = e.sum()
				ope = abs(e).sum()
			case 'pred_err_l2':
				ope = np.linalg.norm(e)
			case 'pw_pred_err':
				ope = np.linalg.norm(e * self.corpus_stats['pi_id'])
			case 'pred_err_wd':
				if self.font == 'word' or self.noise!=0.0:
					ope = np.nan
				else:
					ope = otfuns.get_w(
						s = x.reshape(self.array_dims),
						t = self.corpus_stats['mu'].reshape(self.array_dims))
			case 'pred_err_gwd':
				if self.font == 'word' or self.noise!=0.0:
					ope = np.nan
				else:
					ope = otfuns.get_gw(
						s = x.reshape(self.array_dims),
						t = self.corpus_stats['mu'].reshape(self.array_dims))
			case 'mahalanobis':
				if np.size(self.corpus_stats['pi'])==1 and np.isnan(self.corpus_stats['pi']):
					ope = np.nan
				else:
					ope = (e @ self.corpus_stats['pi'] @ e.T)**.5
			case 'kalmanw_pred_err':
				if np.size(self.corpus_stats['kal'])==1 and np.isnan(self.corpus_stats['kal']):
					ope = np.nan
				else:
					ope = np.linalg.norm(self.corpus_stats['kal'] @ e)

		return ope
	
	def __calculate_canvas_dims__(self, input_words=None, pad_w_per_char=8, pad_h=0):
		if not hasattr(self, 'corpus_df'):
			self.__get_corpus__()

		if input_words is None:
			input_words = self.input_words
		
		unique_fonts = set([self.font, self.prior_font])

		font_dims_f = []
		for f_id in unique_fonts:
			font = ImageFont.truetype(self.font_dict[f_id], self.font_size)

			# get max width and height for the input and corpus words
			test_words = set([*input_words, *self.corpus_df['word']])
			pad_w = int(max([len(w) for w in test_words]) * pad_w_per_char)
			font_dims_f.append( np.max([font.getbbox(w, anchor='lt')[2:] for w in test_words], axis=0) + np.array([pad_w, pad_h]) )
		font_dims_f = np.array(font_dims_f)

		# warn if the dimensions mismatch
		if len(np.unique(font_dims_f[:, 0]))>1 or len(np.unique(font_dims_f[:, 1]))>1:
			warnings.warn('Mismatch between dimensions of input font and prior font - will not make any attempt to align the fonts!')

		# get the maximum in each dimension, for all tested fonts
		font_dims = np.max(font_dims_f, axis=0)

		# store in self
		self.canvas_dims = list(font_dims)
		self.array_dims = (font_dims[1], font_dims[0])

		return None

	def __render_text__(self, text, is_prior=False, noise=0.0, show=False):

		if not hasattr(self, 'canvas_dims'):
			self.__calculate_canvas_dims__()

		# set up font
		if is_prior:
			font = ImageFont.truetype(self.font_dict[self.prior_font], self.font_size)
		else:
			font = ImageFont.truetype(self.font_dict[self.font], self.font_size)

		# Rendering text with pillow
		render   = Image.new('L', self.canvas_dims, color=0)
		draw     = ImageDraw.Draw(render)
		text_pos = ((self.canvas_dims[0] - font.getlength(text))/2, -7)
		draw.text(text_pos, text, fill=255, font=font)
		if show: render.show();

		# Applying additive Gaussian noise
		render_array = np.array(render) / 255 # Normalise to r \in [0, 1]
		noise_array  = noise * np.random.randn(*render_array.shape)
		text_array   = (render_array + noise_array).flatten()

		return text_array
	
	def __render_text_at_pos__(self, text, text_pos, anchor=None, show=False):

		if not hasattr(self, 'canvas_dims'):
			self.__calculate_canvas_dims__()

		# set up font		
		font = ImageFont.truetype(self.font_dict[self.font], self.font_size)

		# Rendering text with pillow
		render   = Image.new('L', self.canvas_dims, color=0)
		draw     = ImageDraw.Draw(render)
		draw.text(text_pos, text, anchor=anchor, fill=255, font=font)
		if show: render.show();
	
		text_array = np.array(render).flatten() / 255

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
	
	def __split_word_img_letters__(self, x_2d, word, is_prior=False, use_cross_cor_meth=False, **kwargs):
		# Input x_2d should be a 2d array of the word, with no noise.
		# Returns an image for each detected letter in the word, with zeroes where the other characters were (can preserve the dimensions of the input in each output)

		if use_cross_cor_meth:
			# the result will be approximate, but handles characters that overlap on the x axis

			# get arrays for each character
			char_arrs = [self.__crop_to_content__(self.__render_text__(L, is_prior=is_prior).reshape(self.array_dims))[0] for L in word]

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

				# create the image with only this letter
				# arr_L_i = self.__render_text_at_pos__(word[L_i], text_pos=[xmin+(xmax-xmin)/2, -7], anchor='ma').reshape(self.array_dims)

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

	def load_opes(self, input_words=None):

		if input_words is None:
			input_words = self.input_words

		if os.path.exists(self.opespath):
			print('Loading existing oPE file...')
			opes_df = pd.read_csv(self.opespath)
			# CSV interprets index info as an unnamed column
			opes_df.rename(columns={'Unnamed: 0':'word'}, inplace=True)

			if len(opes_df.word.unique()) != len(set(input_words)) or not set(input_words).issubset(opes_df.word.unique()):
				warnings.warn(f'Loaded oPE file, but mismatch in words!')
		else:
			print(f'Calculating oPE for {len(input_words)} inputs...')
			self.estimate_corpus_stats(weight_by_freq=self.freq_weight)
			opes_df = self.__create_opes_df__(words=input_words)

		return opes_df


class LetterOrthopeEstimator(OrthopeEstimator):
	
	def __init__(self, language, noise, input_words, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None):
		super().__init__(language, font='word', noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, freq_weight=freq_weight, data_label=data_label)

	def __render_text__(self, text, is_prior=False, noise=0.0, show=False):
		# note that is_prior is ignored for LetterOrthopeEstimator

		# Settings
		alphabet = self.alphabet

		max_n_letters = max([len(text)]) if np.isinf(self.n_letters[1]) else self.n_letters[1]

		render_array = np.zeros((max_n_letters, len(alphabet)))
		for cix, c in enumerate(text):
			render_array[cix, alphabet.index(c)] = 1
		
		noise_array = noise * np.random.randn(*render_array.shape)
		text_array  = (render_array + noise_array).flatten()

		return text_array


class OptimalTransportOrthopeEstimator(OrthopeEstimator):

	def __init__(self, language, font, noise, input_words, prior_font=None, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None):
		super().__init__(language, font=font, noise=noise, input_words=input_words, prior_font=prior_font, n_letters=n_letters, freq_perc=freq_perc, freq_weight=freq_weight, data_label=data_label)

		# separate opespath if using the optimal transport estimator
		data_label = '' if data_label is None else f'{data_label}_'
		opespath_prefix = f'{data_label}{language}_{font}_{prior_font}_noise-{noise}_letters-{n_letters[0]}-{n_letters[1]}_freqperc-{freq_perc[0]}-{freq_perc[1]}_freqweight-{freq_weight}_opes_ot'.replace('.','p')  # add "_ot" suffix
		self.opespath = self.savepath / f'{opespath_prefix}.csv'

		assert self.font != 'word', 'LetterOrthopeEstimator() not implemented for OptimalTransportOrthopeEstimator()'

	def __create_opes_df__(self, words, estimates=None, save=True):

		if estimates is None: 
			estimates = ['pred_err_l1', 'pred_err_l2', 
						 'pred_err_wd',
						#  'pred_err_gwd',
						]
		
		n_obs   = 100 if self.noise > 0 else 1
		opes_df = pd.DataFrame(index=words)

		for est in estimates:
			print(f'Computing estimates for {est}')
			for word in tqdm(words):
				opes = [self.__estimate_ope__(word,est) for _ in range(n_obs)]
				opes_df.at[word, est+'_mu']  = np.mean(opes)
				opes_df.at[word, est+'_std'] = np.std(opes)

		if save:
			opes_df.to_csv(self.opespath)

		return opes_df
	
	def estimate_corpus_stats(self, weight_by_freq=True):
		
		print('Rendering corpus...')
		dd, weights = self.__render_corpora__()

		if not weight_by_freq:
			weights = np.ones(weights.shape)

		weights /= np.sum(weights)

		# print('Estimating word-level barycentre...')
		# dd_3d = dd.reshape([-1, self.array_dims[0], self.array_dims[1]])  # reshape to 3d array of word * x * y
		# bc = otfuns.get_w_barycentre(dd_3d, debias=False, weights=weights, reg=0.0005, numItermax=int(1e7))

		# del dd_3d

		print('Getting letters and weights for each slot...')
		dd_spl = [self.__split_word_img_letters__(dd_i.reshape(self.array_dims), word=w_i, is_prior=True) for dd_i, w_i in zip(dd, self.corpus_df['word'])]

		del dd

		lett_slot_3d = []
		for i in range(self.n_letters[1]):
			lett_slot_3d_i = []
			for dd_spl_j in dd_spl:
				if len(dd_spl_j) >= (i + 1):
					lett_slot_3d_i.append(dd_spl_j[i])
			lett_slot_3d.append(np.array(lett_slot_3d_i))

		del dd_spl

		# treat letters as identical if the values correlate very highly

		# use network of booleans for whether each pair of vectors reaches the threshold to cluster and find unique
		# (this is much less efficient, but still better than using corpus-based analysis to get unique letters, as it allows for different letter positions)
		identical_r = 0.99999  # letter images that correlate at least this strongly will be considered identical, and will be averaged - use a very strict cutoff to avoid equating similar but different letters
		letts = []
		letts_weights = []
		for lett_slot_3d_i in lett_slot_3d:
			lett_slot_3d_i_cr, _ = self.__crop_to_content__(lett_slot_3d_i)
			lett_slot_vecs_i = lett_slot_3d_i_cr.reshape(lett_slot_3d_i_cr.shape[0], -1)
			cors_i = np.corrcoef(lett_slot_vecs_i)
			G = nx.from_numpy_array((cors_i >= identical_r).astype(int))  # graph of binary similarities
			components = list(nx.connected_components(G))
			letts.append( np.array([lett_slot_3d_i[np.array(list(c)).astype(int), :, :].mean(axis=0) for c in components]) )
			letts_weights.append( np.array([np.sum(weights[np.array(list(c)).astype(int)]) for c in components]) )
			del lett_slot_vecs_i, cors_i, G, components

		print(f'N unique letters per slot: {[len(L) for L in letts]}')
		letts_weights = [w / w.sum() for w in letts_weights]

		# crop the images of letters to reduce array size
		letts_cr_out = [self.__crop_to_content__(L, L.max(axis=0)) for L in letts]
		letts_cr = [cr_out_i[0] for cr_out_i in letts_cr_out]
		letts_cr_pads = [cr_out_i[1][1:] for cr_out_i in letts_cr_out]  # pad widths for undoing the crops (ignore the first axis, as the barycentres will be 2d)

		print('Estimating within-letter barycentres...')
		bcs_cr = [otfuns.get_w_barycentre(L, debias=False, weights=w, reg=0.0005, numItermax=int(1e7)) for L, w in zip(letts_cr, letts_weights)]

		# now undo the crop to put the barycentres in the original coordinates
		bcs = [np.pad(bc_i, pad_i, mode='constant', constant_values=0.0) for bc_i, pad_i in zip(bcs_cr, letts_cr_pads)]

		# join into a single image
		bcs_joined = np.sum(bcs, axis=0)

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
	
	def __estimate_ope__(self, word, estimate):

		x = self.__render_text__(word, noise=self.noise)
		x_2d = x.reshape(self.array_dims)
		
		if '_wd' in estimate or '_gwd' in estimate:
			x_letts = self.__split_word_img_letters__(x_2d, word=word, is_prior=False)

		e = [x_2d_i - bc_i for x_2d_i, bc_i in zip(x_letts, self.corpus_stats['bcs'])]

		match estimate:
			case 'pred_err_l1':
				# ope = e.sum()
				ope = abs(e).sum()
			case 'pred_err_l2':
				ope = np.linalg.norm(e)
			case 'pred_err_wd':
				if self.noise!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_w(s = L, t = bc_i) for L, bc_i in zip(x_letts, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
			case 'pred_err_gwd':
				if self.noise!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_gw(s = L, t = bc_i) for L, bc_i in zip(x_letts, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
		return ope
	
	def load_opes(self, input_words=None):

		if input_words is None:
			input_words = self.input_words

		if os.path.exists(self.opespath):
			print('Loading existing oPE file...')
			opes_df = pd.read_csv(self.opespath)
			# CSV interprets index info as an unnamed column
			opes_df.rename(columns={'Unnamed: 0':'word'}, inplace=True)

			if len(opes_df.word.unique()) != set(input_words).issubset(opes_df.word.unique()):
				warnings.warn(f'Loaded oPE file, but mismatch in words!')
		else:
			print(f'Calculating optimal transport oPE for {len(input_words)} inputs...')
			self.estimate_corpus_stats(weight_by_freq=self.freq_weight)
			opes_df = self.__create_opes_df__(words=input_words)

		return opes_df
	
class WithinLetterOptimalTransportOrthopeEstimator(OrthopeEstimator):
	# this function is more efficient, but assumes earlier on that mass is only transported within letter slots

	def __init__(self, language, font, noise, input_words, prior_font=None, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None):
		super().__init__(language, font=font, noise=noise, input_words=input_words, prior_font=prior_font, n_letters=n_letters, freq_perc=freq_perc, freq_weight=freq_weight, data_label=data_label)

		# separate opespath if using the optimal transport estimator
		data_label = '' if data_label is None else f'{data_label}_'
		opespath_prefix = f'{data_label}{language}_{font}_{prior_font}_noise-{noise}_letters-{n_letters[0]}-{n_letters[1]}_freqperc-{freq_perc[0]}-{freq_perc[1]}_freqweight-{freq_weight}_opes_wlot'.replace('.','p')  # add "_wlot" suffix
		self.opespath = self.savepath / f'{opespath_prefix}.csv'

		assert self.font != 'word', 'LetterOrthopeEstimator() not implemented for WithinLetterOptimalTransportOrthopeEstimator()'

	def __calculate_canvas_dims__(self, input_words=None, pad_w_per_char=8, pad_h=0):
		# for this class, the canvas dimensions are only ever one character in width
		if not hasattr(self, 'corpus_df'):
			self.__get_corpus__()

		if input_words is None:
			input_words = self.input_words
		
		font = ImageFont.truetype(self.font_dict[self.font], self.font_size)

		# get max width and height for the letters of the input and corpus words
		input_letters = [l for ls in [list(w) for w in input_words] for l in ls]
		corpus_letters = [l for ls in [list(w) for w in self.corpus_df.word] for l in ls]
		test_letters = set([*input_letters, *corpus_letters])
		pad_w = int(pad_w_per_char)
		font_dims = np.max([font.getbbox(w, anchor='lt')[2:] for w in test_letters], axis=0) + np.array([pad_w, pad_h])

		# store in self
		self.canvas_dims = list(font_dims)
		self.array_dims = (font_dims[1], font_dims[0])

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
			[self.__render_text__(sl, noise=self.noise) for sl in slot_letts_i]
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
		
		n_obs   = 100 if self.noise > 0 else 1
		opes_df = pd.DataFrame(index=words)

		for est in estimates:
			print(f'Computing estimates for {est}')
			for word in tqdm(words):
				opes = [self.__estimate_ope__(word,est) for _ in range(n_obs)]
				opes_df.at[word, est+'_mu']  = np.mean(opes)
				opes_df.at[word, est+'_std'] = np.std(opes)

		if save:
			opes_df.to_csv(self.opespath)

		return opes_df
	
	def estimate_corpus_stats(self, weight_by_freq=True):
		
		print('Rendering corpus...')
		dd, word_weights, lett_weights = self.__render_corpora__()
		dd_2d = [[dd_ij.reshape(self.canvas_dims) for dd_ij in dd_i] for dd_i in dd]

		# if weight_by_freq, then the weights will be frequency-weighted...
		if weight_by_freq:
			weights = [w / np.sum(w) for w in word_weights]
		# ...otherwise, use the letter counts (comparable to the other classes)
		else:
			weights = [w / np.sum(w) for w in lett_weights]

		print('Estimating within-letter barycentres...')
		bcs = [otfuns.get_w_barycentre(np.array(L), debias=False, weights=w, reg=0.0005, numItermax=int(1e7)) for L, w in zip(dd_2d, weights)]

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
	
	def __estimate_ope__(self, word, estimate):

		x = [self.__render_text__(L, noise=self.noise) for L in list(word)]
		x_2d = [x_i.reshape(self.array_dims) for x_i in x]

		e = [x_2d_i - bc_i for x_2d_i, bc_i in zip(x_2d, self.corpus_stats['bcs'])]

		match estimate:
			case 'pred_err_l1':
				ope = abs(np.hstack(e)).sum()
			case 'pred_err_l2':
				ope = np.linalg.norm(np.hstack(e))
			case 'pred_err_wd':
				if self.noise!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_w(s = L, t = bc_i) for L, bc_i in zip(x_2d, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
			case 'pred_err_gwd':
				if self.noise!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_gw(s = L, t = bc_i) for L, bc_i in zip(x_2d, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
		return ope
	
	def load_opes(self, input_words=None):

		if input_words is None:
			input_words = self.input_words

		if os.path.exists(self.opespath):
			print('Loading existing oPE file...')
			opes_df = pd.read_csv(self.opespath)
			# CSV interprets index info as an unnamed column
			opes_df.rename(columns={'Unnamed: 0':'word'}, inplace=True)

			if len(opes_df.word.unique()) != set(input_words).issubset(opes_df.word.unique()):
				warnings.warn(f'Loaded oPE file, but mismatch in words!')
		else:
			print(f'Calculating optimal transport oPE for {len(input_words)} inputs...')
			self.estimate_corpus_stats(weight_by_freq=self.freq_weight)
			opes_df = self.__create_opes_df__(words=input_words)

		return opes_df

def run_all_oPEs(language, font, input_words, n_letters=(5, 5), data_label=None):

	# Optimal Transport approach
	if font != 'word':
		for freq_min in min_freq_percs:
			for freq_weight in (True, False):
				gg = OptimalTransportOrthopeEstimator(language=language, font=font, noise=0.0, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight)
				gg.load_opes()

	# Euclidean approach
	for noise in noises:
		for freq_min in min_freq_percs:
			for freq_weight in (True, False):
				if font == 'word':
					gg = LetterOrthopeEstimator(language=language, noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight)
				else:
					gg = OrthopeEstimator(language=language, font=font, noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight)
				gg.load_opes()

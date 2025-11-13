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
# self = OptimalTransportOrthopeEstimator('german', 'courier', 0.0, ['Tisch', 'Lampe'], data_label='test')

class OrthopeEstimator():

	def __init__(self, language, font, noise, input_words, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None):
		data_label_lab = '' if data_label is None else f'{data_label} '
		freq_wt_lab = 'freq-weighted' if freq_weight else 'freq-unweighted'
		print(f'{data_label_lab}{language}, font {font}, noise {noise}, letters {n_letters}, freq% {freq_perc}, {freq_wt_lab}')

		self.alphabet = string.ascii_letters + special + ' '

		self.language    = language
		self.font        = font
		self.noise       = noise
		self.freq_weight = freq_weight
		self.input_words = input_words

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

		data_label = '' if data_label is None else f'{data_label}_'
		opespath_prefix = f'{data_label}{language}_{font}_noise-{noise}_letters-{n_letters[0]}-{n_letters[1]}_freqperc-{freq_perc[0]}-{freq_perc[1]}_freqweight-{freq_weight}_opes'.replace('.','p')
		self.opespath = self.savepath / f'{opespath_prefix}.csv'

		if not os.path.exists(self.savepath): os.makedirs(self.savepath)
		if not os.path.exists(self.datapath): os.makedirs(self.datapath)

	def __create_opes_df__(self, words, estimates=None, save=True):

		if estimates is None: 
			estimates = ['n_pixels_l1', 'n_pixels_l2', 
						 'pred_err_l1', 'pred_err_l2', 'pw_pred_err', 
						 'pw_err_wd',
						#  'pw_err_gwd',
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

	def __render_corpora__(self):
	
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

		# Computing corpus at pixel space assuming identical obs_noise
		dd = np.array([self.__render_text__(word, noise=self.noise) for word in df['word']])
		weights = df['fpmw'].to_numpy()

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
			case 'pw_err_wd':
				if self.font == 'word' or self.noise!=0.0:
					ope = np.nan
				else:
					ope = otfuns.get_w(
						s = x.reshape(self.array_dims),
						t = self.corpus_stats['mu'].reshape(self.array_dims))
			case 'pw_err_gwd':
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

	def __render_text__(self, text, noise=0.0, standardise_length=True, show=False):

		# Settings
		font_size   = 34
		if standardise_length:
			canvas_dims = (int(round(22*len(text))), 36)
		else:
			canvas_dims = (int(round(22*self.n_letters[1])), 36)

		if not hasattr(self, 'array_dims'):
			self.array_dims = (canvas_dims[1], canvas_dims[0])
		
		font_dict   = {'courier'  : self.fontpath / 'couriernew.ttf',
					   'courieri' : self.fontpath / 'couriernewi.ttf',
					   'cambria'  : self.fontpath / 'cambria.ttf',
					   'verdana'  : self.fontpath / 'verdana.ttf',
					   'cambriai' : self.fontpath / 'cambriai.ttf'}

		# Rendering text with pillow
		render   = Image.new('L', canvas_dims, color=0)
		draw     = ImageDraw.Draw(render)
		font     = ImageFont.truetype(font_dict[self.font], font_size)
		text_pos = ((canvas_dims[0] - font.getlength(text))/2, -7)
		draw.text(text_pos, text, fill=255, font=font)
		if show: render.show();

		# Applying additive Gaussian noise
		render_array = np.array(render) / 255 # Normalise to r \in [0, 1]
		noise_array  = noise * np.random.randn(*render_array.shape)
		text_array   = (render_array + noise_array).flatten()

		return text_array
	
	def __get_letter_space_locs__(self, x_2d):
		# Detects the locations of spaces between letters, assuming that there are no breaks along the x axis within glyphs of a width greater than 12 pixels.
		max_xaxis = x_2d.max(axis=0)

		# dummy code the start and end to the max, so they can be used to detect start and end of word (because of the peak-finding algorithm used)
		max_xaxis[0] = max_xaxis.max()
		max_xaxis[len(max_xaxis)-1] = max_xaxis.max()

		# use peak-finding algorithm to get the spaces' locations
		space_centres, _ = sp.signal.find_peaks(-max_xaxis, distance=12)  # minimum distance is assumed to be less than the expected width of a character

		space_locs = space_centres[1:-1]  # now ignore the zeroes at the starts and ends of the words

		assert len(space_locs) >= self.n_letters[0]-1, f'Detected {len(space_locs)} spaces in a word image, but expected the min to be {self.n_letters[0]-1}'
		assert len(space_locs) <= self.n_letters[1]-1, f'Detected {len(space_locs)} spaces in a word image, but expected the max to be {self.n_letters[1]-1}'

		return space_locs
	
	def __split_word_img_letters__(self, x_2d, space_locs=None):
		# Input x_2d should be a 2d array of the word, with no noise.
		# Returns an image for each detected letter in the word, with zeroes where the other characters were (can preserve the dimensions of the input in each output)
		if space_locs is None:
			space_locs = self.__get_letter_space_locs__(x_2d=x_2d)
		space_locs = np.insert(space_locs, 0, 0.0)

		x_2d_spl = []
		for i in range(len(space_locs)):
			x_2d_i = x_2d.copy()

			if i > 0:
				x_2d_i[:, :space_locs[i]] = 0.0

			if i < len(space_locs)-1:
				x_2d_i[:, space_locs[i+1]:] = 0.0
				
			x_2d_spl.append( x_2d_i )

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

			if len(opes_df.word.unique()) != set(input_words).issubset(opes_df.word.unique()):
				warnings.warn(f'Loaded oPE file, but mismatch in words!')
		else:
			print(f'Calculating oPE for {len(input_words)} inputs...')
			self.estimate_corpus_stats(weight_by_freq=self.freq_weight)
			opes_df = self.__create_opes_df__(words=input_words)

		return opes_df


class LetterOrthopeEstimator(OrthopeEstimator):
	
	def __init__(self, language, noise, input_words, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None):
		super().__init__(language, font='word', noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, freq_weight=freq_weight, data_label=data_label)

	def __render_text__(self, text, noise=0.0, show=False):

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

	def __init__(self, language, font, noise, input_words, n_letters=(5, 5), freq_perc=(0, 100), freq_weight=True, data_label=None):
		super().__init__(language, font=font, noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, freq_weight=freq_weight, data_label=data_label)

		# separate opespath if using the optimal transport estimator
		data_label = '' if data_label is None else f'{data_label}_'
		opespath_prefix = f'{data_label}{language}_{font}_noise-{noise}_letters-{n_letters[0]}-{n_letters[1]}_freqperc-{freq_perc[0]}-{freq_perc[1]}_freqweight-{freq_weight}_opes_ot'.replace('.','p')  # add "_ot" suffix
		self.opespath = self.savepath / f'{opespath_prefix}.csv'

		assert self.font != 'word', 'LetterOrthopeEstimator() not implemented for OptimalTransportOrthopeEstimator()'
	
	def estimate_corpus_stats(self, weight_by_freq=True):
		
		print('Rendering corpus...')
		dd, weights = self.__render_corpora__()

		if not weight_by_freq:
			weights = np.ones(weights.shape)

		# print('Estimating word-level barycentre...')
		# dd_3d = dd.reshape([-1, self.array_dims[0], self.array_dims[1]])  # reshape to 3d array of word * x * y
		# bc = otfuns.get_w_barycentre(dd_3d, debias=False, weights=weights, reg=0.0005, numItermax=int(1e7))

		# del dd_3d

		print('Getting letters and weights for each slot...')
		dd_spl = [self.__split_word_img_letters__(dd_i.reshape(self.array_dims)) for dd_i in dd]

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
		lett_slot_vecs = [lett_slot_3d_i.reshape(lett_slot_3d_i.shape[0], -1) for lett_slot_3d_i in lett_slot_3d]
		cors = [np.corrcoef(lett_slot_vecs_i) for lett_slot_vecs_i in lett_slot_vecs]
		del lett_slot_vecs

		# use network of booleans for whether each pair of vectors reaches the threshold to cluster and find unique
		# (this is much less efficient, but still better than using corpus-based analysis to get unique letters, as it allows for different letter positions)
		identical_r = 0.99999  # letter images that correlate at least this strongly will be considered identical, and will be averaged - use a very strict cutoff to avoid equating similar but different letters
		letts = []
		letts_weights = []
		for cors_i, lett_slot_3d_i in zip(cors, lett_slot_3d):
			G = nx.from_numpy_array((cors_i >= identical_r).astype(int))  # graph of binary similarities
			components = list(nx.connected_components(G))
			letts.append( np.array([lett_slot_3d_i[np.array(list(c)).astype(int), :, :].mean(axis=0) for c in components]) )
			letts_weights.append( np.array([np.sum(weights[np.array(list(c)).astype(int)]) for c in components]) )

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
				im = ax.imshow(np.sum(self.corpus_stats['bcs_joined'], axis=0), interpolation='none', cmap='binary')
				divider = make_axes_locatable(ax)
				cax = divider.append_axes('right', size='2.5%', pad=0.1)
				fig.colorbar(im, cax=cax, orientation='vertical')
				stat_lab = stat
		
		ax.set_title(stat_lab)
		return fig, ax
	
	def __estimate_ope__(self, word, estimate):

		x = self.__render_text__(word, noise=self.noise)
		x_2d = x.reshape(self.array_dims)
		e = x_2d - self.corpus_stats['bcs_joined']
		
		if '_wd' in estimate or '_gwd' in estimate:
			x_letts = self.__split_word_img_letters__(x_2d)

		match estimate:
			case 'pred_err_l1':
				# ope = e.sum()
				ope = abs(e).sum()
			case 'pred_err_l2':
				ope = np.linalg.norm(e)
			case 'pw_err_wd':
				if self.noise!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_w(s = L, t = bc_i) for L, bc_i in zip(x_letts, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
			case 'pw_err_gwd':
				if self.noise!=0.0:
					ope = np.nan
				else:
					ope_L = [otfuns.get_w(s = L, t = bc_i) for L, bc_i in zip(x_letts, self.corpus_stats['bcs'])]
					ope = np.sum(ope_L)
		return ope

def run_all_oPEs(language, font, input_words, n_letters=(5, 5), data_label=None):

	for noise in noises:
		for freq_min in min_freq_percs:
			for freq_weight in (True, False):
				if font == 'word':
					gg = LetterOrthopeEstimator(language=language, noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight)
				else:
					gg = OrthopeEstimator(language=language, font=font, noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=[freq_min, 100], data_label=data_label, freq_weight=freq_weight)
				gg.load_opes()

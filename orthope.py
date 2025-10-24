import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import re
import pandas as pd
import scipy
import seaborn as sns
import string
from pathlib import Path
import collections
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

#noises = [1E-10, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
noises  = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]

special  = 'àâäæçéèêëîïôœùûüÿÀÂÄÆÇÉÈÊËÎÏÔŒÙÛÜŸëïöüĳËÏÖÜĲäöüßÄÖÜẞáéíóúñÁÉÍÓÚÑ'  # special characters to include as alphabetic (in addition to string.ascii_letters)

class OrthopeEstimator():

	def __init__(self, language, font, noise, input_words, n_letters=(5, 5), freq_perc=(0, 100), data_label=None):

		self.alphabet = string.ascii_letters + special + ' '

		self.language    = language
		self.font        = font
		self.noise       = noise
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
		self.opespath = self.savepath / f'{data_label}{language}_{font}_noise-{noise}_opes.csv'

		if not os.path.exists(self.savepath): os.makedirs(self.savepath)
		if not os.path.exists(self.datapath): os.makedirs(self.datapath)

	def __create_opes_df__(self, words, estimates=None, save=True):

		if estimates is None: 
			estimates = ['n_pixels_l1', 'n_pixels_l2', 
						 'pred_err_l1', 'pred_err_l2', 'pw_pred_err', 
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
		datafile = self.corppath / f'{corpora[self.language]['file']}'

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
		dd = np.array([self.__render_text__(word) for word in df['word']])

		return dd, df['fpmw']

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
	
	def __plot_2d_from_flat__(self, x_1d, cmap='Greys', **kwargs):
		x_2d = x_1d.reshape((self.canvas_dims[1], self.canvas_dims[0]))
		fig, ax = plt.subplots()
		im = ax.imshow(x_2d, interpolation='none', cmap=cmap, **kwargs)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='2.5%', pad=0.1)
		fig.colorbar(im, cax=cax, orientation='vertical')
		return fig, ax
	
	def __plot_2dstat__(self, stat, log_trans=False, cmap='Greys'):
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
			stat_4d = self.corpus_stats[stat].reshape((self.canvas_dims[1], self.canvas_dims[0], self.canvas_dims[1], self.canvas_dims[0]))

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
				self.__plot_2dstat__(stat, log_trans=False, cmap='Greys')
			case 'sigma':
				self.__plot_4dstat__(stat)
			case 'pi':
				self.__plot_4dstat__(stat, log_trans=True)
			case 'pi_id':
				self.__plot_2dstat__(stat, log_trans=True, cmap='viridis')
			case 'kal':
				self.__plot_4dstat__(stat)

	def __estimate_ope__(self, word, estimate):

		x = self.__render_text__(word)

		if 'n_pixels_' not in estimate:
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

	def __render_text__(self, text, show=False):

		# Settings
		font_size   = 34
		self.canvas_dims = (int(round(22*self.n_letters[1])), 36)
		font_dict   = {'courier'  : self.fontpath / 'couriernew.ttf',
					   'courieri' : self.fontpath / 'couriernewi.ttf',
					   'cambria'  : self.fontpath / 'cambria.ttf',
					   'verdana'  : self.fontpath / 'verdana.ttf',
					   'cambriai' : self.fontpath / 'cambriai.ttf'}

		# Rendering text with pillow
		render   = Image.new('L', self.canvas_dims, color=0)
		draw     = ImageDraw.Draw(render)
		font     = ImageFont.truetype(font_dict[self.font], font_size)
		text_pos = ((self.canvas_dims[0] - font.getlength(text))/2, -7)
		draw.text(text_pos, text, fill=255, font=font)
		if show: render.show();

		# Applying additive Gaussian noise
		render_array = np.array(render) / 255 # Normalise to r \in [0, 1]
		noise_array  = self.noise * np.random.randn(*render_array.shape)
		text_array   = (render_array + noise_array).flatten()

		return text_array

	def load_opes(self):

		if os.path.exists(self.opespath):
			opes_df = pd.read_csv(self.opespath)
			# CSV interprets index info as an unnamed column
			opes_df.rename(columns={'Unnamed: 0':'word'}, inplace=True)
			opes_df.set_index('word', inplace=True)
		else:
			print('OPEs file not found. Computing...')
			self.estimate_corpus_stats()
			opes_df = self.__create_opes_df__(words=self.input_words)

		return opes_df


class LetterOrthopeEstimator(OrthopeEstimator):
	
	def __init__(self, language, noise, input_words, n_letters=(5, 5), freq_perc=(0, 100), data_label=None):
		super().__init__(language, font='word', noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, data_label=data_label)

	def __render_text__(self, text, show=False):

		# Settings
		alphabet = string.ascii_letters + special + ' '

		max_n_letters = max([len(text)]) if np.isinf(self.n_letters[1]) else self.n_letters[1]

		render_array = np.zeros((max_n_letters, len(alphabet)))
		for cix, c in enumerate(text):
			render_array[cix, alphabet.index(c)] = 1
		
		noise_array = self.noise * np.random.randn(*render_array.shape)
		text_array  = (render_array + noise_array).flatten()

		return text_array








def run_all_oPEs(language, font, input_words, n_letters=(5, 5), freq_perc=(0, 100), data_label=None):

	for noise in noises:
		if font == 'word':
			gg = LetterOrthopeEstimator(language=language, noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, data_label=data_label)
		else:
			gg = OrthopeEstimator(language=language, font=font, noise=noise, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, data_label=data_label)
		gg.load_opes()

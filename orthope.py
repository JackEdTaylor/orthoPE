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

	def __init__(self, language, font, noise, input_words, n_letters=(1, np.inf), fpmw=(0, np.inf)):

		self.alphabet = string.ascii_letters + special

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

		if isinstance(fpmw, collections.abc.Iterable) and len(fpmw)==2:
			self.fpmw = fpmw
		else:
			self.fpmw = (fpmw, fpmw)

		self.datapath = Path('data_repository')
		self.corppath = self.datapath / Path('corpora')
		self.savepath = Path('models')
		self.fontpath = Path('fonts')

		preffix = f'{language}_{font}_noise-' + f'{noise:.2g}'.replace('.','p')
		self.opespath = f'{self.savepath}/{preffix}_opes.csv'

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
		df = df.loc[(df.fpmw>=self.fpmw[0]) & (df.fpmw<=self.fpmw[1])]

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
				if np.isnan():
					ope = np.nan
				else:
					ope = np.linalg.norm(e * self.corpus_stats['pi_id'])
			case 'mahalanobis':
				if np.isnan(self.corpus_stats['pi']):
					ope = np.nan
				else:
					ope = (e @ self.corpus_stats['pi'] @ e.T)**.5
			case 'kalmanw_pred_err':
				if np.isnan(self.corpus_stats['kal']):
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
	
	def __init__(self, language, noise, input_words, n_letters=(1, np.inf), fpmw=(0, np.inf)):
		super().__init__(language, font='word', noise=noise, input_words=input_words, n_letters=n_letters, fpmw=fpmw)

	def __render_text__(self, text, show=False):

		# Settings
		alphabet = string.ascii_letters + special + ' '

		max_n_letters = max([len(text)]) if np.isinf(self.nletters[1]) else self.nletters[1]

		render_array = np.zeros((max_n_letters, len(alphabet)))
		for cix, c in enumerate(text):
			render_array[cix, alphabet.index(c)] = 1
		
		noise_array = self.noise * np.random.randn(*render_array.shape)
		text_array  = (render_array + noise_array).flatten()

		return text_array








def run_all_oPEs(language, font, noise, input_words, n_letters=(1, np.inf), fpmw=(0, np.inf)):

	for noise in noises:
		if font == 'word':
			gg = LetterOrthopeEstimator(language=language, noise=noise, input_words=input_words, n_letters=n_letters, fpmw=fpmw)
		else:
			gg = OrthopeEstimator(language=language, font=font, noise=noise, input_words=input_words, n_letters=n_letters, fpmw=fpmw)
		gg.load_opes()


def compute_oPE_RT_correlations(language, font):

	estimates = ['n_pixels_l1', 'n_pixels_l2', 'pred_err_l1', 'pred_err_l2', 
				 'pw_pred_err', 'mahalanobis', 'kalmanw_pred_err']
	
	stats = []
	for nix, noise in enumerate(noises):

		gg = OrthopeEstimator(language, font, noise)
		df = pd.merge(gg.load_opes(), gg.load_data_rt(), on='word')

		for eix, est in enumerate(estimates):
			for c in df['category'].unique():
				sub  = (df['category'] == c)
				x, y = 'rtz_mu', est+'_mu'
				rho, pval = scipy.stats.pearsonr(df[sub][x],df[sub][y])

				stats.append({'estimate': est, 'noise': noise, 'category': c,
							  'pval': pval, 'rho': rho})

	df_st = pd.DataFrame(stats)

	return df_st


def plot_oPE_RT_scatterplots(language, font):

	respath = './results'
	if not os.path.exists(respath): os.makedirs(respath)

	df_st = compute_oPE_RT_correlations(language, font)
	estimates = df_st['estimate'].unique()
	categories = df_st['category'].unique()

	fig, axes = plt.subplots(len(noises), len(estimates))
	colors    = sns.color_palette()
	sns.set(style="whitegrid")

	for nix, noise in enumerate(noises):
		gg = OrthopeEstimator(language, font, noise)
		df = pd.merge(gg.load_opes(), gg.load_data_rt(), on='word')

		for eix, est in enumerate(estimates):
			
			subset = (df_st['estimate']==est) & (df_st['noise']==noise)
			x, y   = 'rtz_mu', est+'_mu'
			ax     = axes[nix, eix]
			legend = 'brief' if nix + eix == 0 else False
			
			sns.scatterplot(df,x=x,y=y,hue='category',ax=ax,legend=legend)

			# Write down stat texts
			for n, cat in enumerate(categories):
				rho  = df_st[subset&(df_st['category']==cat)]['rho'].iloc[0]
				pval = df_st[subset&(df_st['category']==cat)]['pval'].iloc[0]
				ax.text(0.95, 0.95-n*0.08, f'r={rho:.2f} (p={pval:.2g})', 
						color=colors[n], transform=ax.transAxes, 
						ha='right', va='top')

			ax.set_title(f'{est}')
			ax.set_xlabel("RT (z-score)")
			if eix == 0: ax.set_ylabel(f'noise = {noise:.1f}')

	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
						wspace=0.5, hspace=0.5)
	fig.set_size_inches(3.5 * len(estimates), 3 * len(noises))
	plt.savefig(f'{respath}/scatterplots_RT_{language}_{font}', dpi=300)


def plot_oPE_RT_rhos(language, font):

	respath = './results'
	if not os.path.exists(respath): os.makedirs(respath)

	df_st = compute_oPE_RT_correlations(language, font)
	categories = df_st['category'].unique()
	
	fig, axes  = plt.subplots(1, len(categories))
	sns.set(style="white")
	
	alphas = [.0001, .001, 0.01, .05] 

	for cat, ax in zip(categories, axes):
		sub  = (df_st['category'] == cat)
		rho  = df_st[sub].pivot(index='estimate',columns='noise',values='rho')
		pval = df_st[sub].pivot(index='estimate',columns='noise',values='pval')
		sign = pval.map(lambda x: '*' * sum([x < alpha for alpha in alphas]))
		annot = rho.map(lambda x: f'{x:.2f}') + sign
		sns.heatmap(rho,vmin=-.25,vmax=.25,annot=annot,fmt="",cmap='vlag',ax=ax)
		#sns.heatmap(rho,vmin=-.5,vmax=.5,annot=rho,cmap='vlag',ax=ax)
		ax.set_title(cat)
		ax.set_ylabel('')

	plt.subplots_adjust(left=0.06,right=0.98,bottom=0.12,top=0.93,wspace=0.17)
	fig.set_size_inches(1.5 * len(noises) + 16.5, 4) 
	plt.savefig(f'{respath}/correlations_RT_{language}_{font}', dpi=300)

	
def plot_oPE_EEG_corrs(language, font):

	respath = './results'
	if not os.path.exists(respath): os.makedirs(respath)

	colors    = sns.color_palette()
	estimates = ['n_pixels_l1', 'n_pixels_l2', 'pre_err_l1', 'pred_err_l2', 
				 'pw_pred_err', 'mahalanobis', 'kalmanw_pred_err']

	df_eeg = OrthopeEstimator(language, font, noises[0]).load_data_eeg()

	timewinds = list(set([int(col[4:7]) for col in df_eeg.columns]))
	clusters  = list(set([col[8] for col in df_eeg.columns]))

	for timewind in timewinds:
		for cluster in clusters:

			fig, axes = plt.subplots(len(noises), len(estimates))
			sns.set(style="whitegrid")

			dataset = f'EEG-{timewind}-{cluster}'
			corpus  = f'{language}_{font}'

			for nix, noise in enumerate(noises):
				gg = OrthopeEstimator(language, font, noise)
				df = pd.merge(gg.load_opes(), gg.load_data_rt(), on='word')

				for eix, est in enumerate(estimates):

					x, y = 'rtz_mu', est+'_mu'
					ax = axes[nix, eix]

					# Compute stats per group
					stats = {c: dict() for c in df['category'].unique()}
					for c in stats:
						sub = (df['category'] == c)
						rho,pval = scipy.stats.pearsonr(df[sub][x],df[sub][y])
						stats[c]['r'], stats[c]['p'] = rho, pval

					# Scatter plots
					ll = 'brief' if nix + eix == 0 else False
					sns.scatterplot(df,x=x,y=y,hue='category',ax=ax,legend=ll)

					# Write down stats
					for n, c in enumerate(df['category'].unique()):
						x, y = 0.05, 0.95 - n * 0.08
						text = f"r={stats[c]['r']:.2f} (p={stats[c]['p']:.2g})"
						ref  = ax.transAxes
						ax.text(x, y, text,
								transform=ref,
								color=colors[n],
								ha='left',va='top')

					# Polishing plots
					r2_str=f'(R²={stats[c]['r']**2:.2f},{stats[c]['r']**2:.2f})'
					ax.set_title(f'{est} {r2_str}')
					ax.set_xlabel("RT (as z-score)")
					
					if eix == 0:
						ax.set_ylabel(f'noise = {noise:.1f}')

			plt.subplots_adjust(left=0.05,  right=0.95, bottom=0.05, top=0.95,
								wspace=0.5, hspace=0.5)
			fig.set_size_inches(25, 18) 
			plt.savefig(f'{respath}/correlations_{dataset}_{corpus}.png',dpi=300)



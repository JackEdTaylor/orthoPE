import orthope
import datahandlers
import glob
import pandas as pd
import numpy as np
import time

def compute_all_models(language, input_words, fonts=None, n_letters=(5, 5), freq_perc=(0, 100), data_label=None):

	if (fonts is None) or (fonts == 'word'):
		fonts = ['word']

	for font in fonts:
		orthope.run_all_oPEs(language=language, font=font, input_words=input_words, n_letters=n_letters, freq_perc=freq_perc, data_label=data_label)


def integrate_models_in_csv(modeldict, savename):

	df = orthope.pd.DataFrame()

	for noise in orthope.noises:

		noisetag = f'noise-{noise:.2g}'.replace('.','p')
		csvs = glob.glob(f'./models/*{noisetag}*.csv')

		for model in modeldict:

			for file in csvs:
				
				font = file.split('_')[1]
				print(f'Reading model {model}-{noisetag}: {font}')

				sub_df = orthope.pd.read_csv(file, index_col=0)
				sub_df = sub_df[[f'{model}_mu']]

				namemap = {f'{model}_mu': f'{font}_{modeldict[model]}_{noisetag}'}
				sub_df.rename(columns=namemap, inplace=True)

				df = orthope.pd.concat([df, sub_df], axis='columns')

	df.to_csv(savename + '.csv', index=True)



fonts = ['courier', 'courieri', 'cambria', 'verdana', 'cambriai']
language = 'german'

# calculate for all German stimuli presented in Gagl et al.
dh = datahandlers.Gagl2020DataHandler(language=language)
unique_words = dh.get_unique_words()
nletters_lims = dh.get_nletter_lims()

unique_words = unique_words[:10]

compute_all_models(language, input_words=unique_words, fonts='word', n_letters=nletters_lims, data_label='gagl2020')
compute_all_models(language, input_words=unique_words, fonts=fonts, n_letters=nletters_lims, data_label='gagl2020')


modeldict = {'pred_err_l2':      'PE2', 
			 'mahalanobis':      'Mahalanobis', 
			 'kalmanw_pred_err': 'Kalman'}

integrate_models_in_csv(modeldict, 'all_models')


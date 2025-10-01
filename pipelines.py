import orthope
import glob
import time


def compute_all_models(language, fonts=None):

	if (fonts is None) or (fonts == 'word'):
		fonts = ['word']

	for font in fonts:
		orthope.run_all_oPEs(language, font)
		orthope.plot_oPE_RT_scatterplots(language, font)
		orthope.plot_oPE_RT_rhos(language, font)



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



language = 'german'
fonts = ['courier'] # 'courieri', 'cambria', 'verdana', 'cambriai']

compute_all_models(language, fonts)
compute_all_models(language, 'word')


modeldict = {'pred_err_l2':      'PE2', 
			 'mahalanobis':      'Mahalanobis', 
			 'kalmanw_pred_err': 'Kalman'}

integrate_models_in_csv(modeldict, 'all_models')


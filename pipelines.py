import orthope
import datahandlers
import glob
import pandas as pd
import numpy as np
import time

def compute_all_models(language, input_words, fonts=None, n_letters=(5, 5), data_label=None):

	if (fonts is None) or (fonts == 'word'):
		fonts = ['word']

	for font in fonts:
		orthope.run_all_oPEs(language=language, font=font, input_words=input_words, n_letters=n_letters, data_label=data_label)


fonts = ['courier', 'courieri', 'cambria', 'verdana', 'cambriai']
language = 'german'

# calculate for all German stimuli presented in Gagl et al.
dh = datahandlers.Gagl2020DataHandler(language=language)
unique_words = dh.get_unique_words()
nletters_lims = dh.get_nletter_lims()

unique_words = unique_words[:10]

compute_all_models(language, input_words=unique_words, fonts='word', n_letters=nletters_lims, data_label='gagl2020')
compute_all_models(language, input_words=unique_words, fonts=fonts, n_letters=nletters_lims, data_label='gagl2020')

print('done!')

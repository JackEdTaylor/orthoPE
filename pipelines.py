import orthope
import datahandlers

n_jobs = -2  # use one fewer than max jobs

fonts = ['courier', 'courieri', 'cambria', 'verdana', 'cambriai', 'comic']
language = 'german'

# calculate for all German stimuli presented in Gagl et al.
dh = datahandlers.Gagl2020DataHandler(language=language)
input_words = dh.get_unique_words()
nletters_lims = dh.get_nletter_lims()

orthope.run_all_oPEs(language=language, input_words=input_words, n_letters=nletters_lims, data_label='gagl2020')

print('done!')

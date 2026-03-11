import orthope
import datahandlers
import sys

if len(sys.argv)>1:
    subset = [int(sys.argv[1]), int(sys.argv[2])]
    n_jobs = int(sys.argv[3])
else:
    subset = None
    n_jobs = -1  # use all available cores

language = 'german'

# calculate for all German stimuli presented in Gagl et al.
dh = datahandlers.Gagl2020DataHandler(language=language)
nletters_lims = dh.get_nletter_lims()

orthope.run_all_oPEs(language=language, input_words=input_words, n_letters=nletters_lims, data_label='gagl2020', n_jobs=n_jobs, save_at_each=True, joblib_backend='loky', subset=subset)

print('done!')

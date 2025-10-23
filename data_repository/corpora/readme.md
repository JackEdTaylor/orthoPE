# SUBTLEX Datasets

This directory contains all the SUBTLEX datasets we use, in tab-delimited format. We use consistent column names here:
* *word*: individual "words" (not lemmas)
* *raw_freq* (this column is not always present): raw counts of how many times a word is observed in subtitles
* *fpmw*: frequency per million words in subtitles

## Data sources

* `SUBTLEX-DE.tsv` contains raw frequency and frequency per million words from the raw [SUBTLEX-DE](https://osf.io/py9ba/) dataset. SUBTLEX-DE is distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). More info is provided by Brysbaert et al. ([2011](https://doi.org/10.1027/1618-3169/a000123)).
* `SUBTLEX-FR.tsv` contains frequency per million words from [Lexique383](http://www.lexique.org/) (the *freqfilms2*) variable. Lexique is distribtued under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). More info provided by New et al. ([2007](https://doi.org/10.1017/S014271640707035X)).
* `SUBTLEX-NL.tsv` contains raw frequency and frequency per million words from the [SUBTLEX-NL](https://osf.io/3d8cx/) dataset. SUBTLEX-PL is distributed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/). More info is provided by Keuleers et al. ([2010](http://doi.org/10.3758/BRM.42.3.643)).
* `SUBTLEX-US.tsv` contains raw frequency and frequency per million words from the raw [SUBTLEX-US](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus) dataset. SUBTLEX-US is distributed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/). More info is provided by Brysbaert and New ([2009](https://doi.org/10.3758/BRM.41.4.977)).

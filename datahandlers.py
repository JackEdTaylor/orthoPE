import os
import pandas as pd
import numpy as np
from pathlib import Path

class Gagl2020DataHandler():
	
    def __init__(self, language):
        self.datapath = Path('data_repository')
        self.language = language
        if not os.path.exists(self.datapath): os.makedirs(self.datapath)

    def load_beh_words(self):

        data_dict = {
            'german': {
                'file'    : 'ger_tr_e1_v01.csv',
                'subject' : 'vp',
                'word'    : 'string',
                'category': 'category',
                'catmap': {
                    'Word':             'word',
                    'Pseudoword':       'pseudoword',
                    'Consonant string': 'consonants'}},
            'english': {
                'file'    : 'blp_tr_data_v01.csv',
                'subject' : 'participant',
                'word'    : 'spelling',
                'category': 'lexicality',
                'catmap': {
                    'W': 'word',
                    'N': 'pseudoword'}},
            'french': {
                'file'    : 'flp_tr_data_v01.csv',
                'subject' : 'vp',
                'word'    : 'mot',
                'category': 'lexicality',
                'catmap': {
                    'mot':    'word',
                    'nonmot': 'pseudoword'}}
            }
        
        if self.language not in data_dict: return np.array([])

        data = data_dict[self.language]
        keys = ['subject','word','category']

        df = pd.read_csv(f'{self.datapath}/{data['file']}', encoding='utf-8')
        df = df.rename(columns={data[key]: key for key in keys})
        # df['category'] = df['category'].map(data['catmap'])
        
        self.beh_words = np.unique(df.word.to_numpy(str))

    def load_eeg_words(self):

        data_dict = {'german': {
                        'subject' : 'subject',
                        'word'    : 'string',
                        'category': 'cond',
                        'catmap': {
                            'Words':             'word',
                            'Pseudowords':       'pseudoword',
                            'Consonant strings': 'consonants'}
                        }
                    }
        
        if self.language not in data_dict: return np.array([])

        data = data_dict[self.language]		
        keys = ['subject','word','category']

        twindows = [230, 340, 430]

        twords = []

        for twindow in twindows:

            df = pd.read_csv(self.datapath / f'EEG_{twindow}.csv', encoding='utf-8')
            df = df.rename(columns={data[key]: key for key in keys})
            twords.append(np.unique(df.word.to_numpy(str)))

        self.eeg_words = np.unique(np.concat(twords))

    def get_unique_words(self):
        if not hasattr(self, 'beh_words'):
            self.load_beh_words()
        if not hasattr(self, 'eeg_words'):
            self.load_eeg_words()
        
        self.unique_words = np.sort(np.unique( np.concat([self.beh_words, self.eeg_words]) ))

        return self.unique_words

    def get_nletter_lims(self):
        if not hasattr(self, 'unique_words'):
            self.get_unique_words()

        str_lens = np.strings.str_len(self.unique_words)

        return (str_lens.min(), str_lens.max())
        
    
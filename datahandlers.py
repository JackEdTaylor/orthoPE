import os
import pandas as pd
from pathlib import Path

class DataLoader():
	
    def __init__(self, language):
        self.datapath = Path('data_repository')
        self.language = language
        if not os.path.exists(self.datapath): os.makedirs(self.datapath)

    def read_rt_data_from_csv(self):

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

        data = data_dict[self.language]
        keys = ['subject','word','category']

        df = pd.read_csv(f'{self.datapath}/{data['file']}')
        df = df.rename(columns={data[key]: key for key in keys})
        df['category'] = df['category'].map(data['catmap'])
        
        return df

    def load_data_rt(self):

        df = self.read_rt_data_from_csv()

        # Z-score RTs within each participant
        z_scoring = lambda x: (x - x.mean()) / x.std()
        df['rt_z'] = df.groupby('subject')['rt'].transform(z_scoring)
        df['log_rt'] = df['rt'].transform(np.log)

        # Compute mean RT per word across participants
        rt_mu   = df.groupby('word')['rt'].mean().reset_index()
        rt_std  = df.groupby('word')['rt'].std().reset_index()
        rtz_mu  = df.groupby('word')['rt_z'].mean().reset_index()
        rtz_std = df.groupby('word')['rt_z'].std().reset_index()
        log_rt  = df.groupby('word')['log_rt'].mean().reset_index()
        
        # Rename columns
        rt_mu.rename(columns={'rt': 'rt_mu'}, inplace=True)
        rt_std.rename(columns={'rt': 'rt_std'}, inplace=True)
        rtz_mu.rename(columns={'rt_z': 'rtz_mu'}, inplace=True)
        rtz_std.rename(columns={'rt_z': 'rtz_std'}, inplace=True)
        
        # Merge estimations and add word categories
        word_categories = df[['word', 'category']].drop_duplicates()
        opes     = df[['word', 'ope']].drop_duplicates()
        opes_all = df[['word', 'ope_all']].drop_duplicates()
        rt_df = pd.merge(rt_mu, rt_std,  on='word')
        rt_df = pd.merge(rt_df, rtz_mu,  on='word')
        rt_df = pd.merge(rt_df, rtz_std, on='word')
        rt_df = pd.merge(rt_df, log_rt,  on='word')
        rt_df = pd.merge(rt_df, word_categories, on='word')
        rt_df = pd.merge(rt_df, opes,     on='word')
        rt_df = pd.merge(rt_df, opes_all, on='word')

        # Set 'word' as index
        rt_df.set_index('word', inplace=True)

        return rt_df

    def load_data_eeg(self):

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

        data = data_dict[self.language]		
        keys = ['subject','word','category']

        datapath = './data_repository'
        twindows = [230, 340, 430]

        tmp_df = pd.read_csv(f'{self.datapath}/EEG_{twindows[0]}.csv')
        eeg_df = pd.DataFrame({'word': tmp_df[data['word']].unique()})

        for twindow in twindows:

            df = pd.read_csv(f'{datapath}/EEG_{twindow}.csv')
            df = df.rename(columns={data[key]: key for key in keys})

            # Z-score voltages within each participant
            z_scoring = lambda x: (x - x.mean()) / x.std()
            df['mv_z'] = df.groupby('subject')['mv'].transform(z_scoring)

            # Average across participants

            for cluster in ['P', 'C']:
                subset = (df.cluster==cluster)
                for stat in ['mu', 'std']:

                    field = f'mvz_{twindow}_{cluster}_{stat}'

                    if stat == 'mu':
                        df_tmp = df[subset].groupby('word')['mv_z'].mean()
                    if stat == 'std':
                        df_tmp = df[subset].groupby('word')['mv_z'].std()
                    
                    df_tmp = df_tmp.reset_index()
                    df_tmp.rename(columns={'mv_z': field}, inplace=True)
                    eeg_df = pd.merge(eeg_df, df_tmp)

        eeg_df.set_index('word', inplace=True)

        return eeg_df
    
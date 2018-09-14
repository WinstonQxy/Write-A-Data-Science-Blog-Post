import io
import requests
import json
from html.parser import HTMLParser
import time       

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Imputer

class MyHTMLParser(HTMLParser): 
    ''' Used for the parsing of data retrieved from the Airbnb website. The data is extracted from teh section "bootstrapData" '''  

    def resetData(self):
        self.__data = []

    def handle_data(self, p_data):
        self.__data.append(p_data)
        
    def displayDataBoot(self, p_data, p_keys = [], p_level = 1):        
        if type(p_data) != dict:
            print(f'*** Path: {p_keys}')
            print(f'{p_data[:100]}...')
            return
        
        if p_level > 10:
            return
        
        for key, value in p_data.items():
            if key not in [ 'nearby_cities', 'og_tags', 'android_alternate_url', 'android_deeplink',
                            'canonical_url', 'iphone_deeplink', 'breadcrumb_details', 'neighborhood_breadcrumb_details',
                            'twitter_tags', 'property_search_url', 'rel_image_src', 'other_property_types' ]:
                v_keys = p_keys.copy()
                v_keys.append(key)
                self.displayDataBoot(value, v_keys, p_level = p_level + 1)
        return
        
    def displayData(self, p_display = True):
        v_data = pd.DataFrame(self.__data)
        v_data[0] = v_data[0].apply(lambda x: x.strip()) 
        v_boot = json.loads(v_data.iloc[(v_data[0].str.contains('<!--{"bootstrapData":')).idxmax(), 0][4:-3])
        
        if 'homePDP' in v_boot['bootstrapData']['reduxData'].keys():        
            self.__data = (v_boot['bootstrapData']['reduxData']['homePDP']['listingInfo']['listing']['seo_features'] )        
            if p_display: self.displayDataBoot(self.__data)        
            return 1
        
        return 0
                
def getURL(p_url, p_display = True):
    ''' Used to get the data from a given URL.
               - p_url - URL adrress to be retrieved
               - p_display - display data parser 
        Returns parsed data. ''' 
    print(f'*****************************************************\n{p_url}')
    v_session = requests.session()
    v_req = v_session.get(p_url)
    v_parser = MyHTMLParser()
    v_parser.resetData()
    v_parser.feed(v_req.text)
    v_return = v_parser.displayData(p_display)    
    v_session.cookies.clear()
    v_session.close() 
    return v_return
    
def displayUrlRow(p_data):
    v_display = ''
    for column in p_data.index:        
        v_display += f'{column}: {p_data[column]}; '
    print(v_display)   
    
def displayPriceBins(p_data, p_column, p_valuesInBin = 100):
    ''' Used to display the bins for the prices.
               - p_data - price data
               - p_column - column name to be used for the binning
               - p_valuesInBin - number of units to be included in a bin ''' 
    v_price = p_data[[p_column]].dropna().astype(int).copy()
    v_bins = np.array(range(p_valuesInBin, v_price[p_column].max(), p_valuesInBin))
    v_price[f'{p_column} bins'] = np.digitize(v_price[p_column], v_bins, right = True)    
    
    v_priceGrp = ( v_price.groupby(f'{p_column} bins')
                          .agg({p_column: ['count', 'min', 'max']}) )
    v_priceGrp.columns = ['_'.join(item) for item in v_priceGrp.columns]
    v_count = v_priceGrp[f'{p_column}_count'].sum()
    v_priceGrp[f'{p_column} %'] = v_priceGrp[f'{p_column}_count'] / v_count
    
    print(f"Total number of properties: {v_count}. Bins every {p_valuesInBin} units.")    
    plt.figure(figsize=(20, 6))
    sns.countplot(x = f'{p_column} bins', data = v_price)
    plt.show()
    display(v_priceGrp)
    
def compWMPrice(p_row, p_period, p_period_base, p_coef):
    if p_row[f'{p_period}_price'] == 0: 
        p_row[f'comp {p_period}_price'] = 0
        return p_row    
    
    p_row[f'comp {p_period}_price'] = p_row[p_period_base] * p_coef
    return p_row

def compWMPriceDiff(p_data, p_period):
    ''' Compute weekly / monthly price difference.
               - p_data - price data
               - p_period - type of perriod for the calculation ''' 
    def innerCompWMPrice(p_row):
        v_period_base = 'weekly_price' if p_period == 'monthly' else 'price'
        v_coef = 4 if p_period == 'monthly' else 7
        return compWMPrice(p_row, p_period, v_period_base, v_coef)
    v_price = p_data[['price', 'weekly_price', 'monthly_price']].fillna(0).astype(int).copy()
    v_price = v_price.fillna(0).apply(innerCompWMPrice, axis = 1)
    
    if p_period == 'monthly':
        v_idx = v_price[v_price[f'weekly_price'] == 0].index
        v_price.loc[v_idx, f'comp {p_period}_price'] = v_price.loc[v_idx,'price'] * 30
    
    v_price[f'{p_period}_price diff'] = v_price[f'{p_period}_price'] / v_price[f'comp {p_period}_price']
    
    showWMPriceDistribution(v_price, p_period)
    
    p_data[f'{p_period}_price diff'] = v_price[f'{p_period}_price diff'].apply(lambda x: np.NaN if 0 else x)
    
    return

def showWMPriceDistribution(p_data, p_period):
    ''' Show weekly / monthly price distribution.
               - p_data - price data
               - p_period - type of perriod for the calculation ''' 
    v_data = p_data[p_data[f'{p_period}_price'] != 0].copy()
    print(f"Total number of properties: {v_data.shape[0]}")
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax = sns.distplot(v_data[f'{p_period}_price diff'], ax = ax, color = "g", kde = False)
    # Creating another Y axis
    second_ax = ax.twinx()
    #Plotting kde without hist on the second Y axis
    sns.kdeplot(v_data[f'{p_period}_price diff'], ax = second_ax, color = "k", lw = 2)
    #Removing Y ticks from the second axis
    second_ax.set_yticks([])
    plt.show()
    
    v_price = ( pd.DataFrame(pd.cut(v_data[f'{p_period}_price diff'], [0, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 3, 6]))
                     .groupby(f'{p_period}_price diff')
                     .agg({f'{p_period}_price diff': ['count']}) ).T    
    display(v_price)
    
    return


def executePCA(p_data, p_scaler, p_percent = 0.9):
    ''' Execute the PCA.
               - p_data - data to be used
               - p_scaler - scaler to be applied, can be None
               - p_percent - percentage for the PCA calculation ''' 
    v_pca = PCA(p_percent)
    v_data = Imputer().fit_transform(p_data)    
    if not p_scaler is None:
        v_data = p_scaler.fit_transform(v_data)
    X_pca = v_pca.fit_transform(v_data)
    
    return v_pca, X_pca
        
def pca_results(p_pca, p_columns):   
    ''' Returns the PCA components.
               - p_pca - PCA already fitted
               - p_columns - initial columns in the dataframe '''  
    v_idx = ['Dimension {}'.format(idx + 1) for idx in range(len(p_pca.components_))]
    
    # PCA components
    v_comp = pd.DataFrame(np.round(p_pca.components_, 4), columns = p_columns)
    v_comp.index = v_idx
    
    # PCA explained variance
    v_ratios = p_pca.explained_variance_ratio_.reshape(len(p_pca.components_), 1)
    v_variance = pd.DataFrame(np.round(v_ratios, 4), columns = ['Component Variance'])
    v_variance.index = v_idx
    
    v_comp = v_comp.merge(v_variance, left_index = True, right_index = True)
    v_idx = v_comp.index.values
    for idx in range(len(v_idx)):
        if idx == 0:
            v_comp.loc[v_idx[idx], 'Cumulated Variance'] = v_comp.loc[v_idx[idx], 'Component Variance']
        else:
            v_comp.loc[v_idx[idx], 'Cumulated Variance'] = v_comp.loc[v_idx[idx], 'Component Variance'] \
                                                             + v_comp.loc[v_idx[idx - 1], 'Cumulated Variance']
                
    return v_comp

def getCompWeight(p_results, p_comp):
    ''' Get PCA components weigths.
               - p_results - Results returned by the execution of pocedure pca_results
               - p_comp - number of components to be included '''  
    v_df = pd.DataFrame(p_results.loc[p_results.index[p_comp - 1]])
    v_df['weight'] = v_df[v_df.columns.values[0]].apply(np.abs)  
    v_df.loc['Component Variance', 'weight'] = 1
    v_df.loc['Cumulated Variance', 'weight'] = 1
    return pd.DataFrame(v_df.sort_values('weight', ascending = False)[v_df.columns.values[0]])

def displayPCAResults(p_data, p_scaler, p_colName, p_percent = 0.9, p_comp_no = 6, p_no = 6, p_figsize = 15):
    ''' Displays the PCA results.
               - p_data - data to be used
               - p_scaler - scaler to be applied, can be None
               - p_percent - percentage for the PCA calculation
               - p_comp_no - number of dimensions to be returned
               - p_no - number of components per dimension to be returned
               - p_figsize - figure size '''  
    v_pca, X_pca = executePCA(p_data, p_scaler, p_percent)

    v_results = pca_results(v_pca, p_data.columns.values)
    v_display = v_results['Cumulated Variance'].copy().reset_index()
    fig, ax = plt.subplots(figsize = (16, 6))
    plt.plot( v_display.index.values + 1, 
              v_display['Cumulated Variance'], 
              marker = 'o', markersize = 10 )
    plt.grid(True)
    plt.show()
    
    v_data = getCompWeight(v_results, 1)
    for idx in range(2, p_comp_no + 1): 
        v_data = v_data.merge(getCompWeight(v_results, idx), how = "left", left_index = True, right_index = True)
    
    display(v_data.fillna(0).head(2))
    
    v_pcaComp = v_data.iloc[2:p_no, :]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, p_figsize))
    mask = np.zeros(v_pcaComp.shape)
    mask[v_pcaComp < 0] = 1
    sns.heatmap( v_pcaComp, annot = True, cmap = "Blues", vmin = 0.05, vmax = 0.30, 
                 linewidths=.5, ax = ax1, robust = True, cbar = False, mask = mask )

    mask = np.zeros(v_pcaComp.shape)
    mask[v_pcaComp > 0] = 1
    sns.heatmap( v_pcaComp.apply(np.abs), annot = True, cmap = "Reds", vmin = 0.05, vmax = 0.30, 
                 linewidths=.5, ax = ax2, robust=True, cbar = False, mask = mask )
    ax1.xaxis.tick_top()
    ax2.xaxis.tick_top()
    ax2.set_yticks([])
    plt.show()

    X_pca = pd.DataFrame(X_pca, columns = ['{} {}'.format(p_colName, idx + 1) for idx in range(len(v_pca.components_))] )
    
    return X_pca, v_data
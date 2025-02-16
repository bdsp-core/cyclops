import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from cycler import cycler
line_cycler   = (cycler(color=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
plt.rc("axes", prop_cycle=line_cycler)

def init_results_dataframe_bonobo():
    # load results
    df = pd.read_csv('../Models/generalized_all_ref_loc/results_localized.csv')
    # load location labels
    df_locations = pd.read_csv('../Data/tables/locations_13FEB24.csv')
    df_locations = df_locations.rename(columns={'location':'SpikeLocation'})
    df = df.merge(df_locations[['event_file','SpikeLocation']],on='event_file',how='left')
    return df

def init_locations(dataset):
    if dataset =='BonoboLocal':
        SpikeLocations = ['frontal','central','parietal', 'occipital','temporal','general']
    elif dataset =='Clemson':
        SpikeLocations = ['frontal','central','parietal', 'occipital','temporal']
    else:
        raise ValueError('please specifiy correct dataset to init locations!')
    ChannelLocations = ['frontal','central', 'parietal', 'occipital','temporal']
    return SpikeLocations,ChannelLocations

def get_preds_and_labels(df,SpikeLocation,ChannelLocation):
    # filter for positive and negative samples
    filter_SpikeLocation = (df.SpikeLocation == SpikeLocation) 
    filter_ChannelLocation = (df.ChannelLocation==ChannelLocation)
    filter_agreement = (df.label>=7/8)
    pos = df[filter_SpikeLocation & filter_ChannelLocation & filter_agreement]
    neg = df[(df.label==0)&(df.ChannelLocation==ChannelLocation)][:len(pos)]
    df_eval = pd.concat([pos,neg])    
    labels = df_eval.label.round(0).astype(int)
    preds = df_eval.pred.to_list()
    return labels, preds

def get_results_SpikeLocation(results,SpikeLocation):
    SpikeResults = results[results.SpikeLocation==SpikeLocation]#.sort_values('roc_auc',ascending=False)
    return SpikeResults

def get_results_ChannelLocation(results_SpikeLocation,ChannelLocation):
    row = results_SpikeLocation[results_SpikeLocation.ChannelLocation==ChannelLocation].iloc[0]
    fpr, tpr, roc_auc, N= row[['fpr', 'tpr', 'roc_auc', 'N']]
    return fpr,tpr,roc_auc, N

def plot_subplot(ax,fpr,tpr,roc_auc,N,ChannelLocation):
    ax.plot(fpr,tpr,label=ChannelLocation+f': {roc_auc:.2f}')
    return ax
  
def calcualte_roc_results(df,SpikeLocations,ChannelLocations):
    results={'SpikeLocation':[],'ChannelLocation':[],'fpr':[], 'tpr':[], 'roc_auc':[],'N':[]}
    for SpikeLocation in SpikeLocations:
        for ChannelLocation in ChannelLocations:
            labels,preds = get_preds_and_labels(df,SpikeLocation,ChannelLocation)
            fpr, tpr, thresholds = roc_curve(labels, preds)

            roc_auc = auc(fpr, tpr)
            results['SpikeLocation'].append(SpikeLocation)        
            results['ChannelLocation'].append(ChannelLocation)
            results['fpr'].append(fpr)
            results['tpr'].append(tpr)
            results['roc_auc'].append(roc_auc)
            results['N'].append(len(labels))
    return pd.DataFrame(results)

def get_dataframe(dataset):
    df = pd.read_csv(os.path.join(path_model,f'results_localized_{dataset}.csv'))
    if dataset =='BonoboLocal':
        df_locations = pd.read_csv('../tables/locations_13FEB24.csv')
        df_locations = df_locations.rename(columns={'location':'SpikeLocation'})
    if dataset =='Clemson':
        df_locations = pd.read_csv('../tables/lut_clemson_with_loc.csv')
    df = df.merge(df_locations[['event_file','SpikeLocation']],on='event_file',how='left')
    return df 

def set_params(ax,SpikeLocation):
    ax.set_title(f'{SpikeLocation.capitalize()} spikes, N={N}')
    ax.set_aspect('equal')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.legend(frameon=False)

if __name__ == '__main__':
    path_model = '../Models/generalized_all_ref_loc'
    dataset = 'Clemson'
    df = get_dataframe(dataset)    
    SpikeLocations,ChannelLocations = init_locations(dataset)

    results = calcualte_roc_results(df,SpikeLocations,ChannelLocations)

    fig, axs = plt.subplots(2,3,figsize = (10,7),sharex=True,sharey=True)

    for i,SpikeLocation in enumerate(SpikeLocations):
        ax = axs[i//3,i%3]
        results_SpikeLocation = get_results_SpikeLocation(results,SpikeLocation)
        for j, ChannelLocation in enumerate(results_SpikeLocation.ChannelLocation):
            fpr, tpr, roc_auc, N = get_results_ChannelLocation(results_SpikeLocation,ChannelLocation)
            ax=plot_subplot(ax,fpr,tpr,roc_auc,N,ChannelLocation.capitalize())
        set_params(ax,SpikeLocation)
    axs[0,0].set_ylabel('Tpr')
    axs[1,0].set_ylabel('Tpr')
    for j in range(3):
        axs[1,j].set_xlabel('Fpr')

    fig.tight_layout()
    fig.savefig(os.path.join(path_model,f'channelwise_{dataset}.png'))


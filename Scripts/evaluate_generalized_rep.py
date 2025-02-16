import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from cycler import cycler
line_cycler   = (cycler(color=["#56B4E9","#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
plt.rc("axes", prop_cycle=line_cycler)

path_model = '../Models/generalized_referential/'
df = pd.read_csv(os.path.join(path_model,'results_deleted.csv'))

fig = plt.figure()
x = np.arange(0,20)
y = df.groupby('n_keeper').mean('AUROC').AUROC 
yerr = df.groupby('n_keeper').AUROC.std()
plt.errorbar(x,y,yerr,label='generalized referential',ecolor='#56B4E9',color='#56B4E9')
print('>>> the specialized AUCs are hardcoded <<<')
'''plt.scatter(19,0.965,label='All bipolar',marker='^')
plt.scatter(6,0.933,label='Six bipolar',marker='^')
plt.scatter(2,0.899,label='Two referential')
plt.scatter(19,0.931,label='All referential')
plt.scatter(6,0.910,label='Six referential')
'''
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0.5,1.01,0.05))
plt.grid(alpha=0.2)
plt.xlabel('number of channels')
plt.ylabel('AUROC')
plt.ylim((0.49,1))
plt.legend(frameon=False)
fig.savefig(os.path.join(path_model,'non_localized_task.png'),bbox_inches='tight')

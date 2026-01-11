import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator







#### Results on Natural Dataset ###



for name in ['table_results_']:

  dfs = []
  for runid in range(3):
    tmp = pd.read_csv('./../'+name+str(runid)+'.csv', index_col=False)
    dfs.append(tmp)

  df = pd.concat(dfs)

  df.rename(columns={"Unnamed: 0": "Metric"}, inplace=True)

  by_row_index = df.groupby(df.Metric, sort=False)
  means_df = by_row_index.mean()
  means_df =  means_df.drop(labels=['Time'])
  natural_data = means_df









### Results on Resnet Model with Manually Set Weights ###


for name in ['table_results_']:

  dfs = []
  for runid in range(3):
      dfs.append(pd.read_csv('./../../resnet/'+name+'True_'+str(runid)+'.csv', index_col=False))

  df = pd.concat(dfs)

  df.rename(columns={"Unnamed: 0": "Metric"}, inplace=True)

  by_row_index = df.groupby(df.Metric, sort=False)
  means_df = by_row_index.mean()
  means_df =  means_df.drop(labels=['Time'])
  renet_manual_data = means_df











### Final Results ####
if True:

  renet_manual_data2 = renet_manual_data.copy().drop(labels="Conciseness")
  natural_data2 = natural_data.copy().drop(labels="Conciseness")

  renet_manual_data2 = renet_manual_data2.drop(columns="Lime")

  natural_data2.drop("_soft_PR_abs", inplace=True) # These are the errorous versions of the soft PR, RE and F1 metrics without conversion of GTs to binary. The correct versions of these metrics (without leading _ in their name) are included in the MAD computation
  natural_data2.drop("_soft_RE_abs", inplace=True)
  natural_data2.drop("_soft_F1_abs", inplace=True)
  natural_data2.drop("_soft_PR_pos", inplace=True)
  natural_data2.drop("_soft_RE_pos", inplace=True)
  natural_data2.drop("_soft_F1_pos", inplace=True)
  natural_data2.drop("_soft_PR_neg", inplace=True)
  natural_data2.drop("_soft_RE_neg", inplace=True)
  natural_data2.drop("_soft_F1_neg", inplace=True)

  renet_manual_data2.drop("_soft_PR_abs", inplace=True)
  renet_manual_data2.drop("_soft_RE_abs", inplace=True)
  renet_manual_data2.drop("_soft_F1_abs", inplace=True)
  renet_manual_data2.drop("_soft_PR_pos", inplace=True)
  renet_manual_data2.drop("_soft_RE_pos", inplace=True)
  renet_manual_data2.drop("_soft_F1_pos", inplace=True)
  renet_manual_data2.drop("_soft_PR_neg", inplace=True)
  renet_manual_data2.drop("_soft_RE_neg", inplace=True)
  renet_manual_data2.drop("_soft_F1_neg", inplace=True)


  natural_data2.drop("Atten_C", inplace=True)
  renet_manual_data2.drop("Atten_C", inplace=True)
  
  natural_data2.drop("soft_PR_abs", inplace=True)
  renet_manual_data2.drop("soft_PR_abs", inplace=True)





  # Check alignment axes. I.e, does renet_manual_data2 have the same axes as natural_data2?
  for i in range(len(renet_manual_data2.axes[0])):
    if not renet_manual_data2.axes[0][i] == natural_data2.axes[0][i]:
      print("ERROR",renet_manual_data2.axes[0][i],natural_data2.axes[0][i])

  for i in range(len(renet_manual_data2.axes[1])):
    if not renet_manual_data2.axes[1][i] == natural_data2.axes[1][i]:
      print("ERROR",renet_manual_data2.axes[1][i],natural_data2.axes[1][i])


  # Among all metrics, not centered around the row means

  a_numpy = renet_manual_data2.to_numpy().copy()
  b_numpy = natural_data2.to_numpy().copy()

  print("renet_manual_data2's shape:",a_numpy.shape, "natural_data2's shape:",b_numpy.shape)

  abs_differences = pd.DataFrame(index = natural_data2.axes[0],
                    columns = natural_data2.axes[1],
                    data = np.abs(a_numpy-b_numpy))

  mae = np.mean(abs_differences.to_numpy().copy().flatten())

  mae_notCentered_notCorrected_notOurs = mae


  plt.figure(figsize=(5.8, 7.8))
  g = sns.heatmap(abs_differences,cmap=sns.color_palette("Greys", as_cmap=True),  annot=False,  square=False)
  g.set(xlabel ="Attribution Methods", ylabel = "Metrics", title ='')
  plt.tight_layout()
  plt.savefig('./abs_differences_0_new.eps', format='eps')
  plt.savefig('./abs_differences_0_new.png', format='png')

  # Among our metrics only, not centered around the row means

  abs_differences_our = pd.DataFrame(index = natural_data2.axes[0],
                    columns = natural_data2.axes[1],
                    data = np.abs(a_numpy-b_numpy))

  abs_differences_our = abs_differences_our.loc['cpa!=':]


  mae_our = np.mean(abs_differences_our.to_numpy().copy().flatten())

  mae_notCentered_notCorrected_Ours = mae_our



  # Among all metrics, centered around the row means

  for row in range(a_numpy.shape[0]):
    a_numpy[row,:] = a_numpy[row,:] - np.mean(a_numpy[row,:])
    b_numpy[row,:] = b_numpy[row,:] - np.mean(b_numpy[row,:])

  abs_differences = pd.DataFrame(index = natural_data2.axes[0],
                    columns = natural_data2.axes[1],
                    data = np.abs(a_numpy-b_numpy))

  mae = np.mean(abs_differences.to_numpy().copy().flatten())
  mae_Centered_notCorrected_notOurs = mae


  plt.figure(figsize=(5.8, 7.8))
  g = sns.heatmap(abs_differences,cmap=sns.color_palette("Greys", as_cmap=True),  annot=False,  square=False) # vmin=0,  vmax=1,
  g.set(xlabel ="Attribution Methods", ylabel = "Metrics", title ='')
  plt.tight_layout()
  plt.savefig('./abs_differences_meancentered_new.eps', format='eps')
  plt.savefig('./abs_differences_meancentered_new.png')




  # Among our metrics only, centered around the row means

  abs_differences_our = pd.DataFrame(index = natural_data2.axes[0],
                    columns = natural_data2.axes[1],
                    data = np.abs(a_numpy-b_numpy))

  abs_differences_our = abs_differences_our.loc['cpa!=':]

  mae_our = np.mean(abs_differences_our.to_numpy().copy().flatten())
  mae_Centered_notCorrected_Ours = mae_our


  print("")

  print("Among our own metrics, the mean absolute difference between the synthetic and natural results after mean-centering (see Figure~\\ref{fig:absdiffmanualtrainedresnet_meancentered}) is $", np.round(mae_Centered_notCorrected_Ours,5),"$.")
  print("The mean absolute difference between the synthetic and natural results without mean-centering (see Figure~\\ref{fig:absdiffmanualtrainedresnet}) is $", np.round(mae_notCentered_notCorrected_Ours,5),"$.")
  print("Among all metrics, the mean absolute difference between the synthetic and natural results after mean-centering is $", np.round(mae_Centered_notCorrected_notOurs,5),"$.")
  print("The mean absolute difference between the synthetic and natural results without mean-centering is $", np.round(mae_notCentered_notCorrected_notOurs,5),"$.")



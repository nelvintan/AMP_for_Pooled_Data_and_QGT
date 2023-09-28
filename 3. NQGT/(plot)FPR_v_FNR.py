import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_SE_multi_threshold():

  output_list = load('FPR_v_FNR.npy')
  output_list1 = output_list[0]
  output_list2 = output_list[1]
  output_list3 = output_list[2]

  mean_final_FPR_list1 = output_list1[0]
  mean_final_FNR_list1 = output_list1[1]
  mean_final_FPR_SE_list1 = output_list1[2]
  mean_final_FNR_SE_list1 = output_list1[3]
  
  mean_final_FPR_list2 = output_list2[0]
  mean_final_FNR_list2 = output_list2[1]
  mean_final_FPR_SE_list2 = output_list2[2]
  mean_final_FNR_SE_list2 = output_list2[3]
  
  mean_final_FPR_list3 = output_list3[0]
  mean_final_FNR_list3 = output_list3[1]
  mean_final_FPR_SE_list3 = output_list3[2]
  mean_final_FNR_SE_list3 = output_list3[3]

  # plotting beta1 sq norm correlation vs delta
  plt.plot(mean_final_FNR_list1, mean_final_FPR_list1, linestyle='solid', marker='.', mfc='none', color='blue', markersize=10, label=r"AMP, $\lambda=0.1$")
  plt.plot(mean_final_FNR_list2, mean_final_FPR_list2, linestyle='solid', marker='.', mfc='none', color='red', markersize=10, label=r"AMP, $\lambda=0.2$")
  plt.plot(mean_final_FNR_list3, mean_final_FPR_list3, linestyle='solid', marker='.', mfc='none', color='green', markersize=10, label=r"AMP, $\lambda=0.3$")
  
  plt.plot(mean_final_FNR_SE_list1, mean_final_FPR_SE_list1, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label=r"SE, $\lambda=0.1$")
  plt.plot(mean_final_FNR_SE_list2, mean_final_FPR_SE_list2, linestyle='None', marker='s', mfc='none', color='red', markersize=10, label=r"SE, $\lambda=0.2$")
  plt.plot(mean_final_FNR_SE_list3, mean_final_FPR_SE_list3, linestyle='None', marker='v', mfc='none', color='green', markersize=10, label=r"SE, $\lambda=0.3$")
  
  plt.xlabel("FNR", fontsize=13)
  plt.ylabel("FPR", fontsize=13)
  plt.legend(loc="upper right", fontsize=13)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.savefig('FPR_v_FNR.pdf', bbox_inches='tight')
  plt.show()

  return

plot_GAMP_v_SE_multi_threshold()
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_SE_multi_delta(p, n_list):

  output_list = load('corr_v_delta.npy')
  output_list1 = output_list[0]
  output_list2 = output_list[1]
  output_list3 = output_list[2]

  mean_final_corr_list1 = output_list1[0]
  mean_final_corr_list_SE1 = output_list1[1]
  SD_final_corr_list1 = output_list1[2]
  SD_final_corr_list_SE1 = output_list1[3]
  
  mean_final_corr_list2 = output_list2[0]
  mean_final_corr_list_SE2 = output_list2[1]
  SD_final_corr_list2 = output_list2[2]
  SD_final_corr_list_SE2 = output_list2[3]
  
  mean_final_corr_list3 = output_list3[0]
  mean_final_corr_list_SE3 = output_list3[1]
  SD_final_corr_list3 = output_list3[2]
  SD_final_corr_list_SE3 = output_list3[3]

  # plotting beta1 sq norm correlation vs delta
  size = len(mean_final_corr_list1)
  delta_list = np.array(n_list) / p
  
  plt.errorbar(delta_list, mean_final_corr_list1, yerr=SD_final_corr_list1, color='blue', ecolor='blue', elinewidth=3, capsize=10, label=r"AMP, $\lambda=0.1$")
  plt.plot(delta_list, mean_final_corr_list_SE1, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label=r"SE, $\lambda=0.1$")
  
  plt.errorbar(delta_list, mean_final_corr_list2, yerr=SD_final_corr_list2, color='red', ecolor='red', elinewidth=3, capsize=10, label=r"AMP, $\lambda=0.2$")
  plt.plot(delta_list, mean_final_corr_list_SE2, linestyle='None', marker='s', mfc='none', color='red', markersize=10, label=r"SE, $\lambda=0.2$")
  
  plt.errorbar(delta_list, mean_final_corr_list3, yerr=SD_final_corr_list3, color='green', ecolor='green', elinewidth=3, capsize=10, label=r"AMP, $\lambda=0.3$")
  plt.plot(delta_list, mean_final_corr_list_SE3, linestyle='None', marker='v', mfc='none', color='green', markersize=10, label=r"SE, $\lambda=0.3$")
  
  plt.xlabel(r"$\delta$", fontsize=13)
  plt.ylabel("Correlation", fontsize=13)
  plt.legend(loc="lower right", fontsize=13)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.savefig('corr_v_delta.pdf', bbox_inches='tight')
  plt.show()

  return

p = 500
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
plot_GAMP_v_SE_multi_delta(p, n_list)
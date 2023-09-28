import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_SE_multi_delta(p, n_list):

  output_list = load('GAMP_v_SE_noisy_2L.npy')

  mean_final_corr_list1 = output_list[0]
  mean_final_corr_list2 = output_list[1]
  mean_final_corr_list3 = output_list[2]
  mean_final_corr_list_SE1 = output_list[3]
  mean_final_corr_list_SE2 = output_list[4]
  mean_final_corr_list_SE3 = output_list[5]
  SD_final_corr_list1 = output_list[6]
  SD_final_corr_list2 = output_list[7]
  SD_final_corr_list3 = output_list[8]
  SD_final_corr_list_SE1 = output_list[9]
  SD_final_corr_list_SE2 = output_list[10]
  SD_final_corr_list_SE3 = output_list[11]

  # plotting beta1 sq norm correlation vs delta
  size = len(mean_final_corr_list1)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_final_corr_list1, yerr=SD_final_corr_list1, color='blue', ecolor='blue', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0.1$")
  plt.plot(delta_list, mean_final_corr_list_SE1, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label=r"SE, $\sigma=0.1$")
  plt.errorbar(delta_list, mean_final_corr_list2, yerr=SD_final_corr_list2, color='red', ecolor='red', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0.3$")
  plt.plot(delta_list, mean_final_corr_list_SE2, linestyle='None', marker='o', mfc='none', color='red', markersize=10, label=r"SE, $\sigma=0.3$")
  plt.errorbar(delta_list, mean_final_corr_list3, yerr=SD_final_corr_list3, color='green', ecolor='green', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0.5$")
  plt.plot(delta_list, mean_final_corr_list_SE3, linestyle='None', marker='o', mfc='none', color='green', markersize=10, label=r"SE, $\sigma=0.5$")
  plt.xlabel(r"$\delta$", fontsize=13)
  plt.ylabel("Correlation", fontsize=13)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.legend(loc="upper left", fontsize=13)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.savefig('noisy_2L.pdf', bbox_inches='tight')
  plt.show()

p = 500
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
plot_GAMP_v_SE_multi_delta(p, n_list)
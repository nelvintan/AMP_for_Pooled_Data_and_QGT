import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_others(p, n_list, input_name, output_name):

  output_list = load(input_name)

  mean_final_corr_list_AMP = output_list[0]
  mean_final_corr_list_BPD = output_list[1]
  SD_final_corr_list_AMP = output_list[2]
  SD_final_corr_list_BPD = output_list[3]

  size = len(mean_final_corr_list_AMP)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_final_corr_list_AMP, yerr=SD_final_corr_list_AMP, color='blue', ecolor='blue', elinewidth=3, capsize=10, label=r"AMP")
  plt.errorbar(delta_list, mean_final_corr_list_BPD, yerr=SD_final_corr_list_BPD, color='red', ecolor='red', elinewidth=3, capsize=10, label=r"BPDN")
  plt.xlabel(r"$\delta$",fontsize=13)
  plt.ylabel("Correlation",fontsize=13)
  plt.legend(loc="lower right",fontsize=13)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.savefig(output_name, bbox_inches='tight')
  plt.show()
  plt.clf()

  return

p = 500
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
input_name1 = "corr_v_delta_diff_algo_nu01.npy"
input_name2 = "corr_v_delta_diff_algo_nu03.npy"
output_name1 = "corr_v_delta_nu01.pdf"
output_name2 = "corr_v_delta_nu03.pdf"

plot_GAMP_v_others(p, n_list, input_name1, output_name1)
plot_GAMP_v_others(p, n_list, input_name2, output_name2)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_SE_multi_delta(p, n_list, input_name, output_name, noisy):

  output_list = load(input_name)

  mean_final_corr_list_AMP = output_list[0]
  SD_final_corr_list_AMP = output_list[1]
  mean_final_corr_list_NP = output_list[2]
  SD_final_corr_list_NP = output_list[3]
  mean_final_corr_list_IHT = output_list[4]
  SD_final_corr_list_IHT = output_list[5]

  size = len(mean_final_corr_list_AMP)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_final_corr_list_AMP, yerr=SD_final_corr_list_AMP, color='blue', ecolor='blue', marker='o', elinewidth=3, capsize=10, label=r"AMP")
  if noisy:
    plt.errorbar(delta_list, mean_final_corr_list_NP, yerr=SD_final_corr_list_NP, color='red', ecolor='red', marker='s', elinewidth=3, capsize=10, label=r"CVX")
  else:
    plt.errorbar(delta_list, mean_final_corr_list_NP, yerr=SD_final_corr_list_NP, color='red', ecolor='red', marker='s', elinewidth=3, capsize=10, label=r"LP")
  plt.errorbar(delta_list, mean_final_corr_list_IHT, yerr=SD_final_corr_list_IHT, color='green', ecolor='green', marker='^', elinewidth=3, capsize=10, label=r"IHT")
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower right", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig(output_name, bbox_inches='tight')
  plt.show()
  plt.clf()

p = 500
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
input_name1 = "GAMP_v_SE_3L_diff_algo_sig0.npy"
input_name2 = "GAMP_v_SE_3L_diff_algo_sig01.npy"
input_name3 = "GAMP_v_SE_3L_diff_algo_sig03.npy"
output_name1 = "AMP_v_others_030303_sig0.pdf"
output_name2 = "AMP_v_others_030303_sig01.pdf"
output_name3 = "AMP_v_others_030303_sig03.pdf"

plot_GAMP_v_SE_multi_delta(p, n_list, input_name1, output_name1, False)
plot_GAMP_v_SE_multi_delta(p, n_list, input_name2, output_name2, True)
plot_GAMP_v_SE_multi_delta(p, n_list, input_name3, output_name3, True)
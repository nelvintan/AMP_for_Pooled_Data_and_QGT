import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_SE_multi_delta(p, n_list, filename, outputname):

  output_list = load(filename)

  mean_final_corr_list1 = output_list[0]
  mean_final_corr_list2 = output_list[1]
  mean_final_corr_list_SE1 = output_list[2]
  mean_final_corr_list_SE2 = output_list[3]
  SD_final_corr_list1 = output_list[4]
  SD_final_corr_list2 = output_list[5]
  SD_final_corr_list_SE1 = output_list[6]
  SD_final_corr_list_SE2 = output_list[7]

  # plotting beta1 sq norm correlation vs delta
  size = len(mean_final_corr_list1)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_final_corr_list1, yerr=SD_final_corr_list1, color='blue', ecolor='blue', elinewidth=3, capsize=10, label=r"AMP (known $\pi$)")
  plt.errorbar(delta_list, mean_final_corr_list2, yerr=SD_final_corr_list2, color='red', ecolor='red', elinewidth=3, capsize=10, label=r"AMP (est $\pi$)")
  plt.xlabel(r"$\delta$", fontsize=13)
  plt.ylabel("Correlation", fontsize=13)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.legend(loc="upper left", fontsize=13)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.savefig(outputname, bbox_inches='tight')
  plt.show()
  plt.clf()

p = 500
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]

filename1 = "GAMP_v_SE_est_pi2.npy"
filename2 = "GAMP_v_SE_est_pi1.npy"
outputname1 = "estimated_pi_shift001.pdf"
outputname2 = "estimated_pi_shift005.pdf"
plot_GAMP_v_SE_multi_delta(p, n_list, filename1, outputname1)
plot_GAMP_v_SE_multi_delta(p, n_list, filename2, outputname2)
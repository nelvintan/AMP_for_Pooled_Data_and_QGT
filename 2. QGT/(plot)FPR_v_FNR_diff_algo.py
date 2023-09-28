import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_LP_multi_threshold(input_name, output_name):

  output_list = load(input_name)

  mean_final_FPR_list = output_list[0]
  mean_final_FNR_list = output_list[1]
  mean_final_FPR_SE_list = output_list[2]
  mean_final_FNR_SE_list = output_list[3]
  mean_final_FPR_list_LP = output_list[4]
  mean_final_FNR_list_LP = output_list[5]

  plt.plot(mean_final_FNR_list, mean_final_FPR_list, linestyle='solid', marker='.', mfc='none', color='blue', markersize=10, label=r"AMP")
  plt.plot(mean_final_FNR_list_LP, mean_final_FPR_list_LP, linestyle='solid', marker='x', mfc='none', color='red', markersize=10, label=r"LP")
  plt.xlabel("FNR",fontsize=13)
  plt.ylabel("FPR",fontsize=13)
  plt.legend(loc="upper right",fontsize=13)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.savefig(output_name, bbox_inches='tight')
  plt.show()
  plt.clf()

  return

input_name1 = "FPR_v_FNR_diff_algo_nu01.npy"
input_name2 = "FPR_v_FNR_diff_algo_nu03.npy"
output_name1 = "FPR_v_FNR_nu01.pdf"
output_name2 = "FPR_v_FNR_nu03.pdf"

plot_GAMP_v_LP_multi_threshold(input_name1, output_name1)
plot_GAMP_v_LP_multi_threshold(input_name2, output_name2)
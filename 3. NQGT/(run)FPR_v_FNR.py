import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import linalg
from numpy.random import normal
from numpy.random import multivariate_normal
from numpy.random import binomial
from numpy.random import uniform
from numpy import save

from scipy.stats import norm
from scipy.stats import multivariate_normal as multivariate_normal_sp
from scipy.linalg import eigh
from scipy.integrate import quad

import cvxpy as cp

# Copied this function over from scipy library
def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps

# Copied this function over from scipy library
def is_pos_semi_def_scipy(matrix):
  s, u = eigh(matrix)
  eps = _eigvalsh_to_eps(s)
  if np.min(s) < -eps:
    print('the input matrix must be positive semidefinite')
    return False
  else:
    return True

def generate_Sigma_0(alpha, delta, beta_bar_mean, beta_bar_var, beta_hat_0_mean, beta_hat_0_var):
  
  Sigma_0 = np.zeros((2, 2))
  
  Sigma_0[0, 0] = beta_bar_var + beta_bar_mean**2
  Sigma_0[0, 1] = beta_bar_mean * beta_hat_0_mean
  Sigma_0[1, 0] = beta_bar_mean * beta_hat_0_mean
  Sigma_0[1, 1] = beta_hat_0_var + beta_hat_0_mean**2

  return (Sigma_0 / delta)
'''
Our GAMP functions below -- note that the inputs Z_k and y_bar will be exchanged
for theta^k_i and y_i in our GAMP algorithm.
'''

def g_k_bayes(Z_k, y_bar, Sigma_k, noise_param):

  def func1(eps_bar):
    part1 = y_bar - eps_bar
    part2 = 1 / (2 * noise_param)
    Z = y_bar - eps_bar
    part3 = multivariate_normal_sp.pdf(np.array([Z, Z_k]), mean=np.zeros(2), cov=Sigma_k, allow_singular=True)
    return part1 * part2 * part3

  def func2(eps_bar):
    part2 = 1 / (2 * noise_param)
    Z = y_bar - eps_bar
    part3 = multivariate_normal_sp.pdf(np.array([Z, Z_k]), mean=np.zeros(2), cov=Sigma_k, allow_singular=True)
    return part2 * part3

  # E_Z_given_Zk_ybar = y_bar
  if is_pos_semi_def_scipy(Sigma_k) == False:
    return np.nan
  num = quad(func1, -1*noise_param, noise_param)[0]
  denom = quad(func2, -1*noise_param, noise_param)[0]
  if denom == 0:
    return np.nan
  E_Z_given_Zk_ybar = num / denom
  E_Z_given_Zk = (Sigma_k[1,0] / Sigma_k[1,1]) * Z_k
  Var_Z_given_Zk = Sigma_k[0,0] - (Sigma_k[1,0]**2) / Sigma_k[1,1]

  output = (E_Z_given_Zk_ybar - E_Z_given_Zk) / Var_Z_given_Zk

  return output

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper(Z_k_and_y_bar, Sigma_k, noise_param):

  Z_k = Z_k_and_y_bar[0]
  y_bar = Z_k_and_y_bar[1]

  return g_k_bayes(Z_k, y_bar, Sigma_k, noise_param)

def compute_c_k(theta_k, r_hat_k, Sigma_k):

  n = len(theta_k)
  part1 = np.dot(theta_k, r_hat_k) / n
  part2 = Sigma_k[1,0] * (np.dot(r_hat_k, r_hat_k) / n)
  output = (part1 - part2) / Sigma_k[1,1]
  
  return output

def f_k_bayes(s, mu_k_b, sigma_k_b, nu):

  pdf1 = norm.pdf(s / sigma_k_b - mu_k_b / sigma_k_b) 
  pdf2 = norm.pdf(s / sigma_k_b)
  output = (nu * pdf1) / (nu * pdf1 + (1-nu) * pdf2)

  return output

def f_k_prime(s, mu_k_b, sigma_k_b, nu):

  pdf1 = norm.pdf((s - mu_k_b) / sigma_k_b) 
  pdf2 = norm.pdf(s / sigma_k_b)
  num = nu * pdf1
  denom = nu * pdf1 + (1 - nu) * pdf2

  factor1 = (-1) * ((s - mu_k_b) / (sigma_k_b**2))
  factor2 = (-1) * (s / (sigma_k_b**2))
  num_deriv = nu * factor1 * pdf1
  denom_deriv = num_deriv + (1-nu) * factor2 * pdf2

  # By quotient rule:
  output = ((num_deriv*denom - num*denom_deriv) / denom) / denom
  
  return output

def MSE(beta, beta_hat):
  
  output = np.mean(np.square(beta_hat - beta))

  return output

def MSE_SE(mu_k_b, sigma_k_b, num_MC_samples, nu):
  '''These are computed from the state evolution parameters'''
  MSE = 0
  G_k_vec = normal(0, 1, num_MC_samples)
  beta_bar_vec = binomial(1, nu, num_MC_samples)
  for sample in range(num_MC_samples):
    G_k = G_k_vec[sample]
    beta_bar = beta_bar_vec[sample]
    s = mu_k_b*beta_bar + sigma_k_b*G_k
    f_k = f_k_bayes(s, mu_k_b, sigma_k_b, nu)
    MSE += (f_k - beta_bar)**2
  MSE = MSE / num_MC_samples

  return MSE

def norm_sq_corr(beta, beta_hat):
  
  num = np.square(np.dot(beta, beta_hat))
  denom = np.square(linalg.norm(beta)) * np.square(linalg.norm(beta_hat))
  
  return num / denom

def norm_sq_corr_SE(mu_k_b, sigma_k_b, num_MC_samples, nu):
  
  part1 = 0
  part2 = 0
  part3 = nu
  G_k_vec = normal(0, 1, num_MC_samples)
  beta_bar_vec = binomial(1, nu, num_MC_samples)
  
  for sample in range(num_MC_samples):
    G_k = G_k_vec[sample]
    beta_bar = beta_bar_vec[sample]
    s = mu_k_b*beta_bar + sigma_k_b*G_k
    f_k = f_k_bayes(s, mu_k_b, sigma_k_b, nu)
    part1 += f_k * beta_bar
    part2 += f_k**2

  part1 = part1 / num_MC_samples
  part2 = part2 / num_MC_samples

  norm_sq_corr = (part1**2) / (part2 * part3)

  return norm_sq_corr

def get_SD(var_corr_list, mean_corr_list, succ_run_list):
  
  num_iter = len(mean_corr_list)
  num_runs = len(var_corr_list)

  SD_list = np.zeros(num_iter)
  for iter in range(num_iter):
    var = 0
    for run in range(num_runs):
      corr = var_corr_list[run][iter]
      if corr > 0:
        var += (corr - mean_corr_list[iter])**2
    var = var / succ_run_list[iter]
    SD_list[iter] = np.sqrt(var)

  return SD_list

def f_k_threshold(s, mu_k_b, sigma_k_b, threshold):
  
  normalized_s = s / mu_k_b
  if normalized_s > threshold:
    return 1
  else:
    return 0

def run_GAMP_threshold(n, p, X, y, alpha, nu, lamb, beta, beta_bar_mean, beta_bar_var, beta_hat_0, threshold, num_iter):

  delta = n / p
  beta_hat_0_mean = beta_bar_mean
  beta_hat_0_var = beta_bar_var
  noise_param = lamb * np.sqrt(1 / (delta * alpha * (1-alpha)))
  
  # GAMP initializations
  r_hat_minus_1 = np.zeros(n)
  b_0 = 1
  Sigma_0 = generate_Sigma_0(alpha, delta, beta_bar_mean, beta_bar_var, beta_hat_0_mean, beta_hat_0_var)
  print('Sigma_0\n',Sigma_0)

  # Storage of GAMP variables from previous iteration
  theta_k = np.zeros(n)
  r_hat_k_minus_1 = r_hat_minus_1
  beta_hat_k = beta_hat_0
  b_k = b_0

  # State evolution parameters
  mu_k_b = 0
  sigma_k_b = 0
  Sigma_k = Sigma_0

  # Storage of the estimates beta_hat
  beta_hat_storage = [] # Stores beta_hat from Bayes-optimal f_k
  beta_hat_storage.append(beta_hat_0)
  beta_hat_storage_threshold = []
  beta_hat_storage_threshold.append(beta_hat_0)

  # Storage of the state evolution param mu_k_B
  mu_k_b_storage = []

  prev_min_corr = 0
  for k in range(num_iter):
    print("=== Running iteration: " + str(k+1) + " ===")
    
    # Computing theta_k
    theta_k = np.dot(X, beta_hat_k) - b_k * r_hat_k_minus_1

    # Computing r_hat_k
    theta_k_and_y = np.concatenate((theta_k[:,None],y[:,None]), axis=1)
    r_hat_k = np.apply_along_axis(g_k_bayes_wrapper, 1, theta_k_and_y, Sigma_k, noise_param)

    if (np.isnan(r_hat_k).any() or np.isinf(r_hat_k).any()):
      print('=== EARLY STOPPAGE ===')
      break
    
    # Computing c_k
    c_k = compute_c_k(theta_k, r_hat_k, Sigma_k)
    
    # Computing beta_k_plus_1
    beta_k_plus_1 = np.dot(X.T, r_hat_k) - c_k * beta_hat_k
    E_X_breve = np.ones((n,p)) * np.sqrt(nu / (n*(alpha*p-nu)))
    X_tilde = X - E_X_breve

    # Computing state evolution for the (k+1)th iteration
    mu_k_plus_1_b = np.dot(r_hat_k, r_hat_k) / n
    sigma_k_plus_1_b = np.sqrt(mu_k_plus_1_b)
    
    # Computing beta_hat_k_plus_1
    beta_hat_k_plus_1 = np.apply_along_axis(f_k_bayes, 1, beta_k_plus_1[:,None], mu_k_plus_1_b, sigma_k_plus_1_b, nu)
    beta_hat_k_plus_1 = beta_hat_k_plus_1.ravel() # unravelling the array.
    beta_hat_k_plus_1_thres = np.apply_along_axis(f_k_threshold, 1, beta_k_plus_1[:,None], mu_k_plus_1_b, sigma_k_plus_1_b, threshold)
    beta_hat_k_plus_1_thres = beta_hat_k_plus_1_thres.ravel() # unravelling the array.

    if np.isnan(beta_hat_k_plus_1).any():
      print('=== EARLY STOPPAGE ===')
      break

    # Computing b_k_plus_1
    # (There is no need write a f_k_prime functions because we will never get to the next iteration 
    # -- recall that we only use a thresholding f_k in the last iteration.)
    b_k_plus_1 = 0
    for j in range(p):
      b_k_plus_1 += f_k_prime(beta_k_plus_1[j], mu_k_plus_1_b, sigma_k_plus_1_b, nu)
    b_k_plus_1 = b_k_plus_1 / n

    # Computing state evolution for the (k+1)th iteration
    Sigma_k_plus_1 = np.zeros((2,2))
    Sigma_k_plus_1[0,0] = Sigma_k[0,0]
    temp_variable = np.dot(beta_hat_k_plus_1, beta_hat_k_plus_1) / p
    Sigma_k_plus_1[0,1] = (temp_variable / delta)
    Sigma_k_plus_1[1,0] = (temp_variable / delta)
    Sigma_k_plus_1[1,1] = (temp_variable / delta)

    # deciding termination of algorithm
    # current_min_corr = norm_sq_corr(beta, beta_hat_k_plus_1)
    # if (prev_min_corr >= current_min_corr):
    #   print('=== EARLY STOPPAGE (performance not improving) ===')
    #   break
    # else:
    #   prev_min_corr = current_min_corr

    # Updating parameters and storing them
    beta_hat_storage.append(beta_hat_k_plus_1)
    beta_hat_k = beta_hat_k_plus_1
    beta_hat_storage_threshold.append(beta_hat_k_plus_1_thres)
    r_hat_k_minus_1 = r_hat_k
    b_k = b_k_plus_1
    mu_k_b_storage.append(mu_k_plus_1_b)
    mu_k_b = mu_k_plus_1_b
    sigma_k_b = sigma_k_plus_1_b
    Sigma_k = Sigma_k_plus_1

    print('mu_k_b\n',mu_k_b) # Under bayes-optimal setting, mu_k_b = sigma_k_b^2
    print('Sigma_k:\n',Sigma_k)

  return beta_hat_storage_threshold, mu_k_b_storage

def run_GAMP_v_SE_multi_threshold(p, n, threshold_list, alpha, nu, lamb, num_iter, num_runs, num_MC_samples):
  
  delta = n / p
  num_thresholds = len(threshold_list)

  mean_final_FPR_list = np.zeros(num_thresholds)
  mean_final_FNR_list = np.zeros(num_thresholds)

  mean_final_FPR_SE_list = np.zeros(num_thresholds)
  mean_final_FNR_SE_list = np.zeros(num_thresholds)

  for threshold_index in range(num_thresholds):
    print("*=== Running for threshold: " + str(threshold_list[threshold_index]) + " ===*")
    threshold = threshold_list[threshold_index]
    final_corr = 0
    for run_num in range(num_runs):
      print('=== Run number: ' + str(run_num + 1) + ' ===')

      np.random.seed(run_num) # so that result is reproducible
      
      beta_bar_mean = nu
      beta_bar_var = nu * (1 - nu)
      beta = binomial(1, nu, p)
      num_def = np.sum(beta)
      num_nondef = p - num_def
      beta_hat_0 = binomial(1, nu, p)

      X = binomial(1, alpha, (n, p))
      eps = uniform(low=-1*lamb*np.sqrt(p), high=lamb*np.sqrt(p), size=n)
      y = np.dot(X, beta) + eps
      y = np.round(y)
      def_prop = np.sum(beta) / p
      y_tilde = (y - alpha*p*def_prop) / np.sqrt(n*alpha*(1-alpha))
      X_tilde = (X - alpha) / np.sqrt(n * alpha * (1 - alpha))

      beta_hat_storage, mu_k_b_storage = run_GAMP_threshold(n, p, X_tilde, y_tilde, alpha, nu, lamb,
                                                            beta, beta_bar_mean, beta_bar_var, beta_hat_0, threshold, num_iter)

      # GAMP
      beta_hat = beta_hat_storage[-1]
      FPR = (np.sum(beta_hat) - np.inner(beta, beta_hat)) / num_nondef
      FNR = ((p - np.inner(beta, beta_hat)) - num_nondef) / num_def
      mean_final_FPR_list[threshold_index] += FPR
      mean_final_FNR_list[threshold_index] += FNR

      # State evolution
      mu_k_b = mu_k_b_storage[-1]
      sigma_k_b = np.sqrt(mu_k_b)
      FPR_SE = 1 - norm.cdf((mu_k_b / sigma_k_b) * threshold)
      FNR_SE = norm.cdf((mu_k_b / sigma_k_b) * (threshold - 1))
      mean_final_FPR_SE_list[threshold_index] += FPR_SE
      mean_final_FNR_SE_list[threshold_index] += FNR_SE

  mean_final_FPR_list = mean_final_FPR_list / num_runs
  mean_final_FNR_list = mean_final_FNR_list / num_runs

  mean_final_FPR_SE_list = mean_final_FPR_SE_list / num_runs
  mean_final_FNR_SE_list = mean_final_FNR_SE_list / num_runs

  print('mean_final_FPR_list\n', mean_final_FPR_list)
  print('mean_final_FNR_list\n', mean_final_FNR_list)

  print('mean_final_FPR_SE_list\n', mean_final_FPR_SE_list)
  print('mean_final_FNR_SE_list\n',mean_final_FNR_SE_list)

  return [mean_final_FPR_list, mean_final_FNR_list, mean_final_FPR_SE_list, mean_final_FNR_SE_list]

delta = 0.3
p = 500
n = int(delta*p)
alpha = 0.5
nu = 0.1
lamb1 = 0.1
lamb2 = 0.2
lamb3 = 0.3
num_iter = 10
num_runs = 10
num_MC_samples = 500
threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

output_list1 = run_GAMP_v_SE_multi_threshold(p, n, threshold_list, alpha, nu, lamb1, num_iter, num_runs, num_MC_samples)
output_list2 = run_GAMP_v_SE_multi_threshold(p, n, threshold_list, alpha, nu, lamb2, num_iter, num_runs, num_MC_samples)
output_list3 = run_GAMP_v_SE_multi_threshold(p, n, threshold_list, alpha, nu, lamb3, num_iter, num_runs, num_MC_samples)
output_list = [output_list1, output_list2, output_list3]
save('FPR_v_FNR', np.array(output_list))
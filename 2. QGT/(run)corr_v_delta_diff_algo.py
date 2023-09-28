import numpy as np
from numpy import linalg
from numpy.random import normal
from numpy.random import multivariate_normal
from numpy.random import binomial
from numpy.random import uniform
from numpy import save

from scipy.stats import norm
from scipy.optimize import linprog

def generate_Sigma_0(delta, beta_bar_mean, beta_bar_var, beta_hat_0_mean, beta_hat_0_var):
  
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

def g_k_bayes(Z_k, y_bar, Sigma_k): 

  E_Z_given_Zk_ybar = y_bar
  E_Z_given_Zk = (Sigma_k[1,0] / Sigma_k[1,1]) * Z_k
  Var_Z_given_Zk = Sigma_k[0,0] - (Sigma_k[1,0]**2) / Sigma_k[1,1]

  output = (E_Z_given_Zk_ybar - E_Z_given_Zk) / Var_Z_given_Zk

  return output

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper(Z_k_and_y_bar, Sigma_k):

  Z_k = Z_k_and_y_bar[0]
  y_bar = Z_k_and_y_bar[1]

  return g_k_bayes(Z_k, y_bar, Sigma_k)

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

def run_GAMP(n, p, X, y, nu, beta, beta_bar_mean, beta_bar_var, beta_hat_0, num_iter):

  delta = n / p
  beta_hat_0_mean = beta_bar_mean
  beta_hat_0_var = beta_bar_var
  
  # GAMP initializations
  r_hat_minus_1 = np.zeros(n)
  b_0 = 1
  Sigma_0 = generate_Sigma_0(delta, beta_bar_mean, beta_bar_var, beta_hat_0_mean, beta_hat_0_var)
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

  # Storage of the estimate beta_hat
  beta_hat_storage = []
  beta_hat_storage.append(beta_hat_0)

  # Storage of the state evolution param mu_k_B
  mu_k_b_storage = []

  prev_min_corr = 0
  for k in range(num_iter):
    print("=== Running iteration: " + str(k+1) + " ===")
    
    # Computing theta_k
    theta_k = np.dot(X, beta_hat_k) - b_k * r_hat_k_minus_1

    # Computing r_hat_k
    theta_k_and_y = np.concatenate((theta_k[:,None],y[:,None]), axis=1)
    r_hat_k = np.apply_along_axis(g_k_bayes_wrapper, 1, theta_k_and_y, Sigma_k)

    if (np.isnan(r_hat_k).any() or np.isinf(r_hat_k).any()):
      print('=== EARLY STOPPAGE ===')
      break
    
    # Computing c_k
    c_k = compute_c_k(theta_k, r_hat_k, Sigma_k)
    
    # Computing beta_k_plus_1
    beta_k_plus_1 = np.dot(X.T, r_hat_k) - c_k * beta_hat_k

    # Computing state evolution for the (k+1)th iteration
    mu_k_plus_1_b = np.dot(r_hat_k, r_hat_k) / n
    sigma_k_plus_1_b = np.sqrt(mu_k_plus_1_b)
    
    # Computing beta_hat_k_plus_1
    beta_hat_k_plus_1 = np.apply_along_axis(f_k_bayes, 1, beta_k_plus_1[:,None], mu_k_plus_1_b, sigma_k_plus_1_b, nu)
    beta_hat_k_plus_1 = beta_hat_k_plus_1.ravel() # unravelling the array.

    if np.isnan(beta_hat_k_plus_1).any():
      print('=== EARLY STOPPAGE ===')
      break

    # Computing b_k_plus_1
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
    current_min_corr = norm_sq_corr(beta, beta_hat_k_plus_1)
    if (prev_min_corr >= current_min_corr):
      print('=== EARLY STOPPAGE (performance not improving) ===')
      break
    else:
      prev_min_corr = current_min_corr

    # Updating parameters and storing them
    beta_hat_storage.append(beta_hat_k_plus_1)
    beta_hat_k = beta_hat_k_plus_1
    r_hat_k_minus_1 = r_hat_k
    b_k = b_k_plus_1
    mu_k_b_storage.append(mu_k_plus_1_b)
    mu_k_b = mu_k_plus_1_b
    sigma_k_b = sigma_k_plus_1_b
    Sigma_k = Sigma_k_plus_1

    print('mu_k_b\n',mu_k_b) # Under bayes-optimal setting, mu_k_b = sigma_k_b^2
    print('Sigma_k:\n',Sigma_k)

  return beta_hat_storage, mu_k_b_storage

def run_LP(n, p, X, y):

  obj = np.ones(p)
  LHS_ineq = np.concatenate((np.eye(p),-np.eye(p)))
  RHS_ineq = np.concatenate((np.ones(p),np.zeros(p)))
  LHS_eq = X
  RHS_eq = y
  opt = linprog(c=obj, A_ub=LHS_ineq, b_ub=RHS_ineq, A_eq=LHS_eq, b_eq=RHS_eq)
  print("Linear program:", opt.message)

  return opt.x

def quantize(beta_hat, threshold):
  result = []
  for entry in beta_hat:
    if entry > threshold:
      result.append(1)
    else:
      result.append(0)
  return np.array(result)

def run_LP_threshold(n, p, X, y, threshold):

  obj = np.ones(p)
  LHS_ineq = np.concatenate((np.eye(p),-np.eye(p)))
  RHS_ineq = np.concatenate((np.ones(p),np.zeros(p)))
  LHS_eq = X
  RHS_eq = y
  opt = linprog(c=obj, A_ub=LHS_ineq, b_ub=RHS_ineq, A_eq=LHS_eq, b_eq=RHS_eq)
  print("Linear program:", opt.message)

  return quantize(opt.x, threshold)

def run_GAMP_v_SE_multi_delta(p, n_list, alpha, nu, num_iter, num_runs, num_MC_samples):
  
  num_deltas = len(n_list)

  mean_final_corr_list_AMP = np.zeros(num_deltas)
  mean_final_corr_list_LP = np.zeros(num_deltas)
  var_final_corr_list_AMP = np.zeros((num_runs, num_deltas))
  var_final_corr_list_LP = np.zeros((num_runs, num_deltas))

  for n_index in range(len(n_list)):
    print("*=== Running for n: " + str(n_list[n_index]) + " ===*")
    n = n_list[n_index]
    delta = n / p
    final_corr = 0
    for run_num in range(num_runs):
      print('=== Run number: ' + str(run_num + 1) + ' ===')

      np.random.seed(run_num) # so that result is reproducible
      
      beta_bar_mean = nu
      beta_bar_var = nu * (1 - nu)
      beta = binomial(1, nu, p)
      beta_hat_0 = binomial(1, nu, p)

      X = binomial(1, alpha, (n, p))
      X_tilde = (X - alpha) / np.sqrt(n * alpha * (1 - alpha))
      y = np.dot(X, beta)
      y_tilde = np.dot(X_tilde, beta)

      beta_hat_storage, mu_k_b_storage = run_GAMP(n, p, X_tilde, y_tilde, nu, beta, beta_bar_mean, beta_bar_var, beta_hat_0, num_iter)
      beta_hat_LP = run_LP(n, p, X, y)

      # GAMP
      beta_hat = beta_hat_storage[-1]
      corr_emp = norm_sq_corr(beta, beta_hat)
      mean_final_corr_list_AMP[n_index] += corr_emp
      var_final_corr_list_AMP[run_num][n_index] = corr_emp

      # LP
      corr_emp = norm_sq_corr(beta, beta_hat_LP)
      mean_final_corr_list_LP[n_index] += corr_emp
      var_final_corr_list_LP[run_num][n_index] = corr_emp

  mean_final_corr_list_AMP = mean_final_corr_list_AMP / num_runs
  mean_final_corr_list_LP = mean_final_corr_list_LP / num_runs

  print('mean_final_corr_list_AMP\n',mean_final_corr_list_AMP)
  print('mean_final_corr_list_LP\n',mean_final_corr_list_LP)

  SD_final_corr_list_AMP = np.sqrt(np.sum(np.square(var_final_corr_list_AMP - mean_final_corr_list_AMP), axis=0) / num_runs)
  SD_final_corr_list_LP = np.sqrt(np.sum(np.square(var_final_corr_list_LP - mean_final_corr_list_LP), axis=0) / num_runs)

  print('SD_final_corr_list_AMP\n',SD_final_corr_list_AMP)
  print('SD_final_corr_list_LP\n',SD_final_corr_list_LP)

  return [mean_final_corr_list_AMP, mean_final_corr_list_LP, SD_final_corr_list_AMP, SD_final_corr_list_LP]
    
p = 500
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
alpha = 0.5
nu = 0.1
num_iter = 10
num_runs = 10
num_MC_samples = 500

output_list = run_GAMP_v_SE_multi_delta(p, n_list, alpha, nu, num_iter, num_runs, num_MC_samples)
save('corr_v_delta_diff_algo_nu01', np.array(output_list))

nu = 0.3

output_list = run_GAMP_v_SE_multi_delta(p, n_list, alpha, nu, num_iter, num_runs, num_MC_samples)
save('corr_v_delta_diff_algo_nu03', np.array(output_list))
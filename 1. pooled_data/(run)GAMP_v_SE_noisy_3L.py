import numpy as np
from numpy import linalg
from numpy.random import multivariate_normal
from numpy.random import normal
from numpy.random import binomial
from numpy.random import uniform
from numpy import save

from scipy.stats import multivariate_normal as multivariate_normal_sp
from scipy.linalg import eigh

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

''' Some helper functions '''

def my_round(a):
  k = 1 - int(np.log10(np.absolute(a[0,0])))
  return np.around(a, k)

def norm_pdf(x, mean, var):
  first_part = 1 / np.sqrt(2 * np.pi * var)
  second_part = np.exp((-1/2) * ((x - mean) ** 2 / var))
  return first_part * second_part

def multi_norm_pdf(x, mean, cov): # multivariate normal distribution
  dimension = len(x)
  first_part = 1 / np.sqrt((2 * np.pi) ** dimension * linalg.det(cov))
  second_part = np.exp((-1/2) * np.dot(np.dot((x - mean).T, linalg.pinv(cov)), (x - mean)))
  return first_part * second_part

def generate_Sigma_0_L3(delta, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov):
  
  Sigma_0 = np.zeros((6, 6))

  # Fill in all the diagonals.
  Sigma_0[0, 0] = B_bar_cov[0, 0] + B_bar_mean[0]**2
  Sigma_0[1, 1] = B_bar_cov[1, 1] + B_bar_mean[1]**2
  Sigma_0[2, 2] = B_bar_cov[2, 2] + B_bar_mean[2]**2
  Sigma_0[3, 3] = B_hat_0_row_cov[0, 0] + B_hat_0_row_mean[0]**2
  Sigma_0[4, 4] = B_hat_0_row_cov[1, 1] + B_hat_0_row_mean[1]**2
  Sigma_0[5, 5] = B_hat_0_row_cov[2, 2] + B_hat_0_row_mean[2]**2

  # Fill in all the off-diagonals (the upper triangle).
  Sigma_0[0, 1] = B_bar_cov[0, 1] + B_bar_mean[0] * B_bar_mean[1]
  Sigma_0[0, 2] = B_bar_cov[0, 2] + B_bar_mean[0] * B_bar_mean[2]
  Sigma_0[0, 3] = B_bar_mean[0] * B_hat_0_row_mean[0]
  Sigma_0[0, 4] = B_bar_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[0, 5] = B_bar_mean[0] * B_hat_0_row_mean[2]

  Sigma_0[1, 2] = B_bar_cov[1, 2] + B_bar_mean[1] * B_bar_mean[2]
  Sigma_0[1, 3] = B_bar_mean[1] * B_hat_0_row_mean[0]
  Sigma_0[1, 4] = B_bar_mean[1] * B_hat_0_row_mean[1]
  Sigma_0[1, 5] = B_bar_mean[1] * B_hat_0_row_mean[2]

  Sigma_0[2, 3] = B_bar_mean[2] * B_hat_0_row_mean[0]
  Sigma_0[2, 4] = B_bar_mean[2] * B_hat_0_row_mean[1]
  Sigma_0[2, 5] = B_bar_mean[2] * B_hat_0_row_mean[2]

  Sigma_0[3, 4] = B_hat_0_row_cov[0, 1] + B_hat_0_row_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[3, 5] = B_hat_0_row_cov[0, 2] + B_hat_0_row_mean[0] * B_hat_0_row_mean[2]

  Sigma_0[4, 5] = B_hat_0_row_cov[1, 2] + B_hat_0_row_mean[1] * B_hat_0_row_mean[2]

  # Fill in all the off-diagonals (the lower triangle).

  Sigma_0[1, 0] = Sigma_0[0, 1]
  Sigma_0[2, 0] = Sigma_0[0, 2]
  Sigma_0[2, 1] = Sigma_0[1, 2]
  Sigma_0[4, 3] = Sigma_0[3, 4]
  Sigma_0[5, 3] = Sigma_0[3, 5]
  Sigma_0[5, 4] = Sigma_0[4, 5]

  Sigma_0[3:, :3] = Sigma_0[:3, 3:].T

  return Sigma_0 / delta

'''
Our GAMP functions below -- note that the inputs Z_k and Y_bar will be exchanged
for Theta^k_i and Y_i in our matrix-GAMP algorithm.
'''

def Var_Z_given_Zk_L3(Sigma_k):
  return Sigma_k[0:3, 0:3] - np.dot(np.dot(Sigma_k[0:3, 3:6], linalg.pinv(Sigma_k[3:6, 3:6])), Sigma_k[3:6, 0:3])

def E_Z_given_Zk_L3(Sigma_k, Z_k):
  return np.dot(np.dot(Sigma_k[0:3, 3:6], linalg.pinv(Sigma_k[3:6, 3:6])), Z_k)

def E_Z_given_Zk_Ybar_L3(Z_k, Y_bar, Sigma_k, Psi_cov):
  cov_matrix = np.zeros((9, 9))
  cov_matrix[:6, :6] = Sigma_k
  cov_matrix[6:, :3] = Sigma_k[:3, :3]
  cov_matrix[:3, 6:] = Sigma_k[:3, :3]
  cov_matrix[6:, 3:6] = Sigma_k[:3, 3:]
  cov_matrix[3:6, 6:] = Sigma_k[:3, 3:]
  cov_matrix[6:, 6:] = Sigma_k[:3, :3] + Psi_cov
  output = np.dot(cov_matrix[:3, 3:], np.dot(linalg.pinv(cov_matrix[3:, 3:]), np.concatenate((Z_k, Y_bar), axis=0)))
  return output

def g_k_bayes_L3(Z_k, Y_bar, Sigma_k, Psi_cov):

  # mat1 = Var_Z_given_Zk_L3(Sigma_k)
  mat1 = my_round(Var_Z_given_Zk_L3(Sigma_k))
  vec2 = E_Z_given_Zk_Ybar_L3(Z_k, Y_bar, Sigma_k, Psi_cov)
  vec3 = E_Z_given_Zk_L3(Sigma_k, Z_k)

  return np.dot(linalg.pinv(mat1), vec2 - vec3)

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper_L3(Z_k_and_Y_bar, Sigma_k, Psi_cov):
  Z_k = Z_k_and_Y_bar[:3]
  Y_bar = Z_k_and_Y_bar[3:]
  return g_k_bayes_L3(Z_k, Y_bar, Sigma_k, Psi_cov)

# Only holds for categorical prior w/ length 3. 
def f_k_bayes_L3(B_bar_k, M_k_B, T_k_B, B_bar_prob):

  numerator = np.zeros(3)
  denomenator = 0

  length = len(B_bar_k)
  for index in range(length):
    b_bar = np.eye(length)[index]
    b_bar_pmf = B_bar_prob[index]
    B_bar_k_pdf = multivariate_normal_sp.pdf(B_bar_k, mean=np.dot(M_k_B, b_bar), cov=T_k_B, allow_singular=True)
    numerator += b_bar * b_bar_pmf * B_bar_k_pdf
    denomenator += b_bar_pmf * B_bar_k_pdf
  
  output = numerator / denomenator

  return output

# Only holds for categorical prior w/ length 2. 
def f_k_bayes_prime_L3(B_bar_k, M_k_B, T_k_B, B_bar_prob):

  num1 = 0 # numerator of {f_k(s)}_1
  num1_deriv = np.zeros(3) # derivative of numerator of {f_k(s)}_1

  num2 = 0 # numerator of {f_k(s)}_2
  num2_deriv = np.zeros(3) # derivative of numerator of {f_k(s)}_2

  num3 = 0 # numerator of {f_k(s)}_3
  num3_deriv = np.zeros(3) # derivative of numerator of {f_k(s)}_3

  denom = 0 # denomenator of {f_k(s)}_1, {f_k(s)}_2, {f_k(s)}_3
  denom_deriv = np.zeros(3) # derivative of denomenator of {f_k(s)}_1, {f_k(s)}_2, {f_k(s)}_3

  length = len(B_bar_k)
  for index in range(length):
    b_bar = np.eye(length)[index]
    b_bar_pmf = B_bar_prob[index]
    mean = np.dot(M_k_B, b_bar)
    B_bar_k_pdf = multivariate_normal_sp.pdf(B_bar_k, mean=mean, cov=T_k_B, allow_singular=True)
    exponent_deriv = np.dot(linalg.pinv(T_k_B), mean - B_bar_k)

    num1 += b_bar[0] * b_bar_pmf * B_bar_k_pdf
    num1_deriv += exponent_deriv * b_bar[0] * b_bar_pmf * B_bar_k_pdf

    num2 += b_bar[1] * b_bar_pmf * B_bar_k_pdf
    num2_deriv += exponent_deriv * b_bar[1] * b_bar_pmf * B_bar_k_pdf

    num3 += b_bar[2] * b_bar_pmf * B_bar_k_pdf
    num3_deriv += exponent_deriv * b_bar[2] * b_bar_pmf * B_bar_k_pdf

    denom += b_bar_pmf * B_bar_k_pdf
    denom_deriv += exponent_deriv * b_bar_pmf * B_bar_k_pdf

  output = np.zeros((3, 3))

  # Apply quotient rule
  row1 = (num1_deriv*denom - num1*denom_deriv) / (denom**2)
  row2 = (num2_deriv*denom - num2*denom_deriv) / (denom**2)
  row3 = (num3_deriv*denom - num3*denom_deriv) / (denom**2)
  output[0, :] = row1 
  output[1, :] = row2
  output[2, :] = row3

  return output

def compute_C_k_L3(Theta_k, R_hat_k, Sigma_k):
  n = len(Theta_k)
  part1 = np.dot(Theta_k.T, R_hat_k)/n
  part2 = np.dot(Sigma_k[3:6,0:3], np.dot(R_hat_k.T, R_hat_k)/n)
  output = np.dot(linalg.pinv(Sigma_k[3:6,3:6]), part1 - part2)
  return output.T

def corr(B, B_hat):
  p = len(B)
  output = 0
  for j in range(p):
    B_hat_j = B_hat[j, :]
    B_j = B[j, :]
    output += np.inner(B_hat_j, B_j)
  output = output / p
  return output

def SE_corr_L3(M_k_B, B_bar_prob, num_MC_samples):

  T_k_B = M_k_B
  indices = np.random.choice(np.array([0, 1, 2]), size=num_MC_samples, p=B_bar_prob)
  B_bar_samples = np.eye(np.max(indices)+1)[indices]
  G_k_B_samples = multivariate_normal([0, 0, 0], T_k_B, num_MC_samples)
  
  output = 0
  for i in range(num_MC_samples):
    B_bar_sample = B_bar_samples[i]
    G_k_B_sample = G_k_B_samples[i]
    s = np.dot(M_k_B, B_bar_sample) + G_k_B_sample
    f = f_k_bayes_L3(s, M_k_B, T_k_B, B_bar_prob)
    f = np.eye(len(f))[f.argmax()]
    output += np.inner(f, B_bar_sample)
  output = output / num_MC_samples

  return output

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

def quantize(B_hat):
  for row in range(len(B_hat)):
    B_hat[row, :] = np.eye(len(B_hat[row]))[B_hat[row].argmax()]
  return B_hat

def run_matrix_GAMP_L3(n, p, X, Y, Psi_cov, B_bar_prob, B, B_bar_mean, B_bar_cov, B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter):

  delta = n / p
  Theta = np.dot(X, B)

  # Matrix-GAMP initializations
  R_hat_minus_1 = np.zeros((n,3))
  F_0 = np.eye(3)

  Sigma_0 = generate_Sigma_0_L3(delta, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov)
  print('Sigma_0\n',Sigma_0)

  # Storage of GAMP variables from previous iteration
  Theta_k = np.zeros((n,3))
  R_hat_k_minus_1 = R_hat_minus_1
  B_hat_k = B_hat_0
  F_k = F_0

  # State evolution parameters
  M_k_B = np.zeros((3,3))
  T_k_B = M_k_B
  Sigma_k = Sigma_0

  # Storage of the estimate B_hat
  B_hat_storage = []
  B_hat_storage.append(B_hat_0)

  # Storage of the state evolution param M_k_B
  M_k_B_storage = []

  prev_min_corr = 0
  for k in range(num_iter):
    print("=== Running iteration: " + str(k+1) + " ===")
    
    # Computing Theta_k
    Theta_k = np.dot(X, B_hat_k) - np.dot(R_hat_k_minus_1, F_k.T)

    # Computing R_hat_k
    Theta_k_and_Y = np.concatenate((Theta_k,Y), axis=1)
    R_hat_k = np.apply_along_axis(g_k_bayes_wrapper_L3, 1, Theta_k_and_Y, Sigma_k, Psi_cov)

    if (np.isnan(R_hat_k).any()):
      print('=== EARLY STOPPAGE ===')
      break
    
    # Computing C_k
    C_k = compute_C_k_L3(Theta_k, R_hat_k, Sigma_k)
    
    # Computing B_k_plus_1
    B_k_plus_1 = np.dot(X.T, R_hat_k) - np.dot(B_hat_k, C_k.T)

    # Computing state evolution for the (k+1)th iteration
    M_k_plus_1_B = np.dot(R_hat_k.T, R_hat_k) / n
    T_k_plus_1_B = M_k_plus_1_B
    
    # Computing B_hat_k_plus_1
    B_hat_k_plus_1 = np.apply_along_axis(f_k_bayes_L3, 1, B_k_plus_1, M_k_plus_1_B, T_k_plus_1_B, B_bar_prob)

    if (np.isnan(B_hat_k_plus_1).any()):
      print('=== EARLY STOPPAGE ===')
      break

    # Computing F_k_plus_1
    F_k_plus_1 = np.zeros((3, 3))
    for j in range(p):
      F_k_plus_1 += f_k_bayes_prime_L3(B_k_plus_1[j], M_k_plus_1_B, T_k_plus_1_B, B_bar_prob)
    F_k_plus_1 = F_k_plus_1 / n

    # Computing state evolution for the (k+1)th iteration
    Sigma_k_plus_1 = np.zeros((6,6))
    Sigma_k_plus_1[0:3,0:3] = Sigma_k[0:3,0:3]
    temp_matrix = np.dot(B_hat_k_plus_1.T, B_hat_k_plus_1) / p
    Sigma_k_plus_1[0:3,3:6] = temp_matrix / delta
    Sigma_k_plus_1[3:6,0:3] = temp_matrix / delta
    Sigma_k_plus_1[3:6,3:6] = temp_matrix / delta

    if (np.isnan(Sigma_k_plus_1).any()):
      print('=== EARLY STOPPAGE ===')
      break

    # Updating parameters and storing B_hat_k_plus_1 & M_k_plus_1_B
    B_hat_storage.append(B_hat_k_plus_1)
    R_hat_k_minus_1 = R_hat_k
    B_hat_k = B_hat_k_plus_1
    F_k = F_k_plus_1
    M_k_B_storage.append(M_k_plus_1_B)
    M_k_B = M_k_plus_1_B
    T_k_B = T_k_plus_1_B
    Sigma_k = Sigma_k_plus_1

    print('M_k_B\n',M_k_B)
    print('Sigma_k:\n',Sigma_k)

  return B_hat_storage, M_k_B_storage

def run_GAMP_v_SE_multi_delta(alpha, p, sigma_vec, n_list, B_bar_prob, B_bar_mean, B_bar_cov, num_iter, num_runs, num_MC_samples):
  
  num_deltas = len(n_list)
  sigma1 = sigma_vec[0]
  sigma2 = sigma_vec[1]
  sigma3 = sigma_vec[2]

  mean_final_corr_list1 = np.zeros(num_deltas)
  var_final_corr_list1 = np.zeros((num_runs, num_deltas))

  mean_final_corr_list2 = np.zeros(num_deltas)
  var_final_corr_list2 = np.zeros((num_runs, num_deltas))

  mean_final_corr_list3 = np.zeros(num_deltas)
  var_final_corr_list3 = np.zeros((num_runs, num_deltas))

  mean_final_corr_list_SE1 = np.zeros(num_deltas)
  var_final_corr_list_SE1 = np.zeros((num_runs, num_deltas))

  mean_final_corr_list_SE2 = np.zeros(num_deltas)
  var_final_corr_list_SE2 = np.zeros((num_runs, num_deltas))

  mean_final_corr_list_SE3 = np.zeros(num_deltas)
  var_final_corr_list_SE3 = np.zeros((num_runs, num_deltas))
  
  B_hat_0_row_mean = B_bar_mean
  B_hat_0_row_cov = B_bar_cov

  for n_index in range(len(n_list)):
    n = n_list[n_index]
    delta = n / p
    final_corr = 0
    for run_num in range(num_runs):
      print('=== Run number: ' + str(run_num + 1) + ' ===')

      np.random.seed(run_num) # so that result is reproducible

      indices = np.random.choice(np.array([0, 1, 2]), size=p, p=B_bar_prob)
      B = np.eye(np.max(indices)+1)[indices]
      indices = np.random.choice(np.array([0, 1, 2]), size=p, p=B_bar_prob)
      B_hat_0 = np.eye(np.max(indices)+1)[indices]

      X = binomial(1, alpha, (n, p))
      X_tilde = (X - alpha) / np.sqrt(n * alpha * (1 - alpha))

      # sigma = 0.1
      tilde_Psi_cov = (1 / (delta * alpha * (1 - alpha))) * np.diag(np.array([sigma1**2, sigma1**2, sigma1**2]))
      noise_matrix = normal(0, np.sqrt((1 / (delta * alpha * (1 - alpha))) * sigma1**2), (n, 3)) 
      Y_tilde = np.dot(X_tilde, B) + noise_matrix 
      B_hat_storage1, M_k_B_storage1 = run_matrix_GAMP_L3(n, p, X_tilde, Y_tilde, tilde_Psi_cov, B_bar_prob, B, B_bar_mean, B_bar_cov, B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter)
      num_iter_ran1 = len(B_hat_storage1)

      # sigma = 0.3
      tilde_Psi_cov = (1 / (delta * alpha * (1 - alpha))) * np.diag(np.array([sigma2**2, sigma2**2, sigma2**2]))
      noise_matrix = normal(0, np.sqrt((1 / (delta * alpha * (1 - alpha))) * sigma2**2), (n, 3)) 
      Y_tilde = np.dot(X_tilde, B) + noise_matrix
      B_hat_storage2, M_k_B_storage2 = run_matrix_GAMP_L3(n, p, X_tilde, Y_tilde, tilde_Psi_cov, B_bar_prob, B, B_bar_mean, B_bar_cov, B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter)
      num_iter_ran2 = len(B_hat_storage2)

      # sigma = 0.5
      tilde_Psi_cov = (1 / (delta * alpha * (1 - alpha))) * np.diag(np.array([sigma3**2, sigma3**2, sigma3**2]))
      noise_matrix = normal(0, np.sqrt((1 / (delta * alpha * (1 - alpha))) * sigma3**2), (n, 3)) 
      Y_tilde = np.dot(X_tilde, B) + noise_matrix
      B_hat_storage3, M_k_B_storage3 = run_matrix_GAMP_L3(n, p, X_tilde, Y_tilde, tilde_Psi_cov, B_bar_prob, B, B_bar_mean, B_bar_cov, B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter)
      num_iter_ran3 = len(B_hat_storage3)

      ''' sigma1 '''
      # GAMP
      B_hat = B_hat_storage1[num_iter_ran1 - 1]
      corr_emp = corr(B, quantize(B_hat))
      mean_final_corr_list1[n_index] += corr_emp
      var_final_corr_list1[run_num][n_index] = corr_emp

      # State evolution
      M_k_B = M_k_B_storage1[num_iter_ran1 - 2] # -2 because there is one less M_k_B than B_hat (due to initialization)
      corr_SE = SE_corr_L3(M_k_B, B_bar_prob, num_MC_samples)
      mean_final_corr_list_SE1[n_index] += corr_SE
      var_final_corr_list_SE1[run_num][n_index] = corr_SE

      ''' sigma2 '''
      # GAMP
      B_hat = B_hat_storage2[num_iter_ran2 - 1]
      corr_emp = corr(B, quantize(B_hat))
      mean_final_corr_list2[n_index] += corr_emp
      var_final_corr_list2[run_num][n_index] = corr_emp

      # State evolution
      M_k_B = M_k_B_storage2[num_iter_ran2 - 2] # -2 because there is one less M_k_B than B_hat (due to initialization)
      corr_SE = SE_corr_L3(M_k_B, B_bar_prob, num_MC_samples)
      mean_final_corr_list_SE2[n_index] += corr_SE
      var_final_corr_list_SE2[run_num][n_index] = corr_SE

      ''' sigma3 '''
      # GAMP
      B_hat = B_hat_storage3[num_iter_ran3 - 1]
      corr_emp = corr(B, quantize(B_hat))
      mean_final_corr_list3[n_index] += corr_emp
      var_final_corr_list3[run_num][n_index] = corr_emp

      # State evolution
      M_k_B = M_k_B_storage3[num_iter_ran3 - 2] # -2 because there is one less M_k_B than B_hat (due to initialization)
      corr_SE = SE_corr_L3(M_k_B, B_bar_prob, num_MC_samples)
      mean_final_corr_list_SE3[n_index] += corr_SE
      var_final_corr_list_SE3[run_num][n_index] = corr_SE

  mean_final_corr_list1 = mean_final_corr_list1 / num_runs
  mean_final_corr_list_SE1 = mean_final_corr_list_SE1 / num_runs

  mean_final_corr_list2 = mean_final_corr_list2 / num_runs
  mean_final_corr_list_SE2 = mean_final_corr_list_SE2 / num_runs

  mean_final_corr_list3 = mean_final_corr_list3 / num_runs
  mean_final_corr_list_SE3 = mean_final_corr_list_SE3 / num_runs

  print('mean_final_corr_list1\n',mean_final_corr_list1)
  print('mean_final_corr_list_SE1\n',mean_final_corr_list_SE1)

  print('mean_final_corr_list2\n',mean_final_corr_list2)
  print('mean_final_corr_list_SE2\n',mean_final_corr_list_SE2)

  print('mean_final_corr_list3\n',mean_final_corr_list3)
  print('mean_final_corr_list_SE3\n',mean_final_corr_list_SE3)

  SD_final_corr_list1 = np.sqrt(np.sum(np.square(var_final_corr_list1 - mean_final_corr_list1), axis=0) / num_runs)
  SD_final_corr_list_SE1 = np.sqrt(np.sum(np.square(var_final_corr_list_SE1 - mean_final_corr_list_SE1), axis=0) / num_runs)

  SD_final_corr_list2 = np.sqrt(np.sum(np.square(var_final_corr_list2 - mean_final_corr_list2), axis=0) / num_runs)
  SD_final_corr_list_SE2 = np.sqrt(np.sum(np.square(var_final_corr_list_SE2 - mean_final_corr_list_SE2), axis=0) / num_runs)

  SD_final_corr_list3 = np.sqrt(np.sum(np.square(var_final_corr_list3 - mean_final_corr_list3), axis=0) / num_runs)
  SD_final_corr_list_SE3 = np.sqrt(np.sum(np.square(var_final_corr_list_SE3 - mean_final_corr_list_SE3), axis=0) / num_runs)

  print('SD_final_corr_list1\n',SD_final_corr_list1)
  print('SD_final_corr_list_SE1\n',SD_final_corr_list_SE1)

  print('SD_final_corr_list2\n',SD_final_corr_list2)
  print('SD_final_corr_list_SE2\n',SD_final_corr_list_SE2)

  print('SD_final_corr_list3\n',SD_final_corr_list3)
  print('SD_final_corr_list_SE3\n',SD_final_corr_list_SE3)

  return [mean_final_corr_list1, mean_final_corr_list2, mean_final_corr_list3, mean_final_corr_list_SE1, mean_final_corr_list_SE2, mean_final_corr_list_SE3, 
          SD_final_corr_list1, SD_final_corr_list2, SD_final_corr_list3, SD_final_corr_list_SE1, SD_final_corr_list_SE2, SD_final_corr_list_SE3]

p = 500
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
num_iter = 10
num_runs = 10
num_MC_samples = 500

alpha = 0.5
sigma_vec = [0.1, 0.3, 0.5]
B_bar_prob = [1/3, 1/3, 1/3]
B_bar_mean = B_bar_prob
B_bar_cov = np.array([
    [B_bar_prob[0]-(B_bar_prob[0])**2, -1*B_bar_prob[0]*B_bar_prob[1], -1*B_bar_prob[0]*B_bar_prob[2]],
    [-1*B_bar_prob[1]*B_bar_prob[0], B_bar_prob[1]-(B_bar_prob[1])**2, -1*B_bar_prob[1]*B_bar_prob[2]],
    [-1*B_bar_prob[2]*B_bar_prob[0], -1*B_bar_prob[2]*B_bar_prob[1], B_bar_prob[2]-(B_bar_prob[2])**2]
    ])

output_list = run_GAMP_v_SE_multi_delta(alpha, p, sigma_vec, n_list, B_bar_prob, B_bar_mean, B_bar_cov, num_iter, num_runs, num_MC_samples)
save('GAMP_v_SE_noisy_3L', np.array(output_list))
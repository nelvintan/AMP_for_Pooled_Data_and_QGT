import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import numpy as np
from numpy import linalg
from numpy.random import multivariate_normal
from numpy.random import normal
from numpy.random import binomial
from numpy.random import uniform
from numpy import save

from scipy.optimize import linprog
from scipy.stats import multivariate_normal as multivariate_normal_sp
from scipy.linalg import eigh
import cvxpy as cp

''' Helper functions '''

def corr(B, B_hat):
  p = len(B)
  output = 0
  for j in range(p):
    B_hat_j = B_hat[j, :]
    B_j = B[j, :]
    output += np.inner(B_hat_j, B_j)
  output = output / p
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

''' Approximate message passing '''

''' Some helper functions '''

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

''' AMP for noisy pooled data with L = 3 '''

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

def g_k_bayes_L3(Z_k, Y_bar, Sigma_k, Psi_cov, noise_present):

  # mat1 = Var_Z_given_Zk(Sigma_k)
  mat1 = my_round(Var_Z_given_Zk_L3(Sigma_k))
  if noise_present:
    vec2 = E_Z_given_Zk_Ybar_L3(Z_k, Y_bar, Sigma_k, Psi_cov)
  else:
    vec2 = Y_bar
  vec3 = E_Z_given_Zk_L3(Sigma_k, Z_k)

  return np.dot(linalg.pinv(mat1), vec2 - vec3)

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper_L3(Z_k_and_Y_bar, Sigma_k, Psi_cov, noise_present):
  Z_k = Z_k_and_Y_bar[:3]
  Y_bar = Z_k_and_Y_bar[3:]
  return g_k_bayes_L3(Z_k, Y_bar, Sigma_k, Psi_cov, noise_present)

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

# Only holds for categorical prior w/ length 3.
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
    output += np.inner(f, B_bar_sample)
  output = output / num_MC_samples

  return output

def run_matrix_GAMP_L3(n, p, X, Y, Psi_cov, B_bar_prob, B, B_bar_mean, B_bar_cov, B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, noise_present):

  delta = n / p
  Theta = np.dot(X, B)
  Y = Theta

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
    try:
      R_hat_k = np.apply_along_axis(g_k_bayes_wrapper_L3, 1, Theta_k_and_Y, Sigma_k, Psi_cov, noise_present)
    except:
      print("=== EARLY STOPPAGE ===")
      break

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

    # deciding termination of algorithm
    current_min_corr = corr(B, B_hat_k_plus_1)
    if (prev_min_corr >= current_min_corr):
      print('=== EARLY STOPPAGE ===')
      break
    else:
      prev_min_corr = current_min_corr

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

''' Optimization methods '''

def run_LP(n, p, L, Y, X, B_prop):

  # Configuring our inputs to suit LP
  Y_LP = Y.flatten('F')
  X_LP = np.zeros((n*L,p*L))
  for l in range(L):
    X_LP[n*l:n*(l+1),p*l:p*(l+1)] = X
  C_LP = np.eye(p)
  for l in range(L-1):
    C_LP = np.concatenate((C_LP, np.eye(p)), axis=1)
  one_p = np.ones(p)

  # Setting the objective vector
  c = np.zeros(p*L)
  for l in range(L):
    I_pL = np.eye(p*L)
    I_pL_trun = I_pL[p*l:p*(l+1),:]
    c -= np.log(B_prop[l]) * np.dot(one_p.T,I_pL_trun)

  # Setting the equality constraints matrix
  A = np.concatenate((X_LP, C_LP), axis=0)
  # Setting the equality constraints vector
  b = np.concatenate((Y_LP, one_p), axis=0)

  # Setting the inequality constraints matrix
  G = np.concatenate((np.eye(p*L), -np.eye(p*L)), axis=0)
  # Setting the inequality constraints vector
  h = np.concatenate((np.ones(p*L), np.zeros(p*L)), axis=0)

  # Solving linear programming problem
  res = linprog(c, A_eq=A, b_eq=b, A_ub=G, b_ub=h)
  print("Linear program:", res.message)

  # Reconfiguring our outputs to suit pooled data
  B_LP_est = res.x
  B_est = np.zeros((p,L))
  for l in range(L):
    B_col = B_LP_est[p*l:p*(l+1)]
    B_est[:,l] = B_col

  return B_est

def run_NP(n, p, L, Y, X, sigma, B_prop):

  # Configuring our inputs to suit LP
  Y_opt = Y.flatten('F')
  X_opt = np.zeros((n*L,p*L))
  for l in range(L):
    X_opt[n*l:n*(l+1),p*l:p*(l+1)] = X
  C_opt = np.eye(p)
  for l in range(L-1):
    C_opt = np.concatenate((C_opt, np.eye(p)), axis=1)
  one_p = np.ones(p)

  # Setting the objective matrix and vector
  q = np.zeros(p*L)
  for l in range(L):
    I_pL = np.eye(p*L)
    I_pL_trun = I_pL[p*l:p*(l+1),:]
    q -= np.log(B_prop[l]) * np.dot(one_p.T,I_pL_trun)

  # Setting the equality constraints matrix
  A = C_opt
  # Setting the equality constraints vector
  b = one_p

  # Setting the inequality constraints matrix
  G = np.concatenate((np.eye(p*L), -np.eye(p*L)), axis=0)
  # Setting the inequality constraints vector
  h = np.concatenate((np.ones(p*L), np.zeros(p*L)), axis=0)

  # Define and solve the CVXPY problem
  constant = 1/(2*p*(sigma**2))
  B_opt = cp.Variable(p*L)
  objective = cp.Minimize(constant*cp.sum_squares(Y_opt - X_opt @ B_opt) + (q.T @ B_opt))
  constraints = []
  constraints.append(G @ B_opt <= h)
  constraints.append(A @ B_opt == b)
  problem = cp.Problem(objective, constraints)
  problem.solve(solver=cp.OSQP)
  print('optimal obj value:', problem.value)

  # Reconfiguring our outputs to suit pooled data
  B_QP_est = B_opt.value
  B_est = np.zeros((p,L))
  for l in range(L):
    B_col = B_QP_est[p*l:p*(l+1)]
    B_est[:,l] = B_col

  return B_est

''' Iterative Hard Thresholding '''

def hard_thres(input, sparsity_lvls):
  L = len(sparsity_lvls)
  output = np.zeros((p, L))
  for l in range(L):
    sparsity_lvl = sparsity_lvls[l]
    input_l = input[:,l]
    top_indices = np.argpartition(input_l,-sparsity_lvl)[-sparsity_lvl:]
    output_l = np.zeros(len(input_l))
    for j in top_indices:
      output_l[j] = input_l[j]
    output[:,l] = output_l
  return output

def IHT(Y, X, sparsity_lvls, num_iter):
  L = len(sparsity_lvls)
  B_k = np.zeros((p, L))
  for k in range(num_iter):
    input = B_k + np.dot(X.T,Y-np.dot(X,B_k))
    B_k_plus_1 = hard_thres(input, sparsity_lvls)
    B_k = quantize(B_k_plus_1)
  return B_k

def one_hot_vec(index, length):
  output = np.zeros(length)
  output[index] = 1
  return output

def category_check(matrix, estimate, sparsity_lvls):
  p = len(estimate)
  L = len(estimate[0])
  output = matrix
  for l in range(L):
    l_num_items = sparsity_lvls[l]*p
    if np.sum(estimate[:,l]) == l_num_items:
      output[:,l] = np.full(p,matrix.min())
  return output

def estimate(input, sparsity_lvls):
  p = len(input)
  L = len(input[0])
  output = np.zeros((p,L))
  matrix = input
  for iter in range(p):
    item, category = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    output[item,:] = one_hot_vec(category, L)
    matrix[item,:] = np.full(L,input.min())
    matrix = category_check(matrix,output,sparsity_lvls)
  return output

# The greedy version
def IHT_greedy(Y, X, sparsity_lvls, num_iter):
  L = len(sparsity_lvls)
  B_k = np.zeros((p, L))
  for k in range(num_iter):
    input = B_k + np.dot(X.T,Y-np.dot(X,B_k))
    B_k = estimate(input, sparsity_lvls)
  return B_k

''' Comparing all 3 algorithms '''

def run_AMP_v_NP_v_IHT_L3_noiseless(alpha, p, n_list, B_bar_prob, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, num_runs, num_MC_samples):

  num_deltas = len(n_list)
  L = len(B_bar_prob)

  mean_final_corr_list_AMP = np.zeros(num_deltas)
  var_final_corr_list_AMP = np.zeros((num_runs, num_deltas))

  mean_final_corr_list_LP = np.zeros(num_deltas)
  var_final_corr_list_LP = np.zeros((num_runs, num_deltas))

  mean_final_corr_list_IHT = np.zeros(num_deltas)
  var_final_corr_list_IHT = np.zeros((num_runs, num_deltas))

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

      # AMP
      X_tilde = (X - alpha) / np.sqrt(n * alpha * (1 - alpha))
      tilde_Psi_cov = np.zeros((3,3)) # matrix not used, created to satisfy inputs to GAMP
      Y_tilde = np.dot(X_tilde, B)
      B_hat_storage= run_matrix_GAMP_L3(n, p, X_tilde, Y_tilde, tilde_Psi_cov, B_bar_prob, B, B_bar_mean, B_bar_cov,
                                        B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, noise_present=False)[0]
      num_iter_ran = len(B_hat_storage)
      B_hat = B_hat_storage[num_iter_ran - 1]
      corr_emp = corr(B, B_hat)
      mean_final_corr_list_AMP[n_index] += corr_emp
      var_final_corr_list_AMP[run_num][n_index] = corr_emp

      # LP
      Y = np.dot(X, B)
      B_prop = np.sum(B, axis=0) / p
      B_hat = run_LP(n, p, L, Y, X, B_prop)
      corr_emp = corr(B, B_hat)
      mean_final_corr_list_LP[n_index] += corr_emp
      var_final_corr_list_LP[run_num][n_index] = corr_emp

      # IHT
      sparsity_lvls = []
      for l in range(L):
        sparsity_lvls.append(int(B_prop[l]*p))
      B_hat = IHT_greedy(Y_tilde, X_tilde, sparsity_lvls, num_iter=10)
      corr_emp = corr(B, B_hat)
      mean_final_corr_list_IHT[n_index] += corr_emp
      var_final_corr_list_IHT[run_num][n_index] = corr_emp

  mean_final_corr_list_AMP = mean_final_corr_list_AMP / num_runs
  SD_final_corr_list_AMP = np.sqrt(np.sum(np.square(var_final_corr_list_AMP - mean_final_corr_list_AMP), axis=0) / num_runs)
  print('mean_final_corr_list_AMP\n',mean_final_corr_list_AMP)
  print('SD_final_corr_list_AMP\n',SD_final_corr_list_AMP)

  mean_final_corr_list_LP = mean_final_corr_list_LP / num_runs
  SD_final_corr_list_LP = np.sqrt(np.sum(np.square(var_final_corr_list_LP - mean_final_corr_list_LP), axis=0) / num_runs)
  print('mean_final_corr_list_NP\n',mean_final_corr_list_LP)
  print('SD_final_corr_list_NP\n',SD_final_corr_list_LP)

  mean_final_corr_list_IHT = mean_final_corr_list_IHT / num_runs
  SD_final_corr_list_IHT = np.sqrt(np.sum(np.square(var_final_corr_list_IHT - mean_final_corr_list_IHT), axis=0) / num_runs)
  print('mean_final_corr_list_IHT\n',mean_final_corr_list_IHT)
  print('SD_final_corr_list_IHT\n',SD_final_corr_list_IHT)

  return [mean_final_corr_list_AMP, SD_final_corr_list_AMP, mean_final_corr_list_LP, SD_final_corr_list_LP,
          mean_final_corr_list_IHT, SD_final_corr_list_IHT]

def run_AMP_v_NP_v_IHT_L3_noisy(alpha, p, sigma, n_list, B_bar_prob, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, num_runs, num_MC_samples):

  num_deltas = len(n_list)
  L = len(B_bar_prob)

  mean_final_corr_list_AMP = np.zeros(num_deltas)
  var_final_corr_list_AMP = np.zeros((num_runs, num_deltas))

  mean_final_corr_list_NP = np.zeros(num_deltas)
  var_final_corr_list_NP = np.zeros((num_runs, num_deltas))

  mean_final_corr_list_IHT = np.zeros(num_deltas)
  var_final_corr_list_IHT = np.zeros((num_runs, num_deltas))

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

      # AMP
      X_tilde = (X - alpha) / np.sqrt(n * alpha * (1 - alpha))
      tilde_Psi_cov = (1 / (delta * alpha * (1 - alpha))) * np.diag(np.array([sigma**2, sigma**2, sigma**2]))
      noise_matrix = normal(0, np.sqrt((1 / (delta * alpha * (1 - alpha))) * sigma**2), (n, L))
      Y_tilde = np.dot(X_tilde, B) + noise_matrix
      B_hat_storage = run_matrix_GAMP_L3(n, p, X_tilde, Y_tilde, tilde_Psi_cov, B_bar_prob, B, B_bar_mean, B_bar_cov,
                                         B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, noise_present=True)[0]
      num_iter_ran = len(B_hat_storage)
      B_hat = B_hat_storage[num_iter_ran - 1]
      corr_emp = corr(B, B_hat)
      mean_final_corr_list_AMP[n_index] += corr_emp
      var_final_corr_list_AMP[run_num][n_index] = corr_emp

      # NP
      Psi = normal(0, sigma, (n, L))
      Y = np.dot(X, B) + Psi
      B_prop = np.sum(B, axis=0) / p
      B_hat = run_NP(n, p, L, Y, X, sigma, B_prop)
      corr_emp = corr(B, B_hat)
      mean_final_corr_list_NP[n_index] += corr_emp
      var_final_corr_list_NP[run_num][n_index] = corr_emp

      # IHT
      sparsity_lvls = []
      for l in range(L):
        sparsity_lvls.append(int(B_prop[l]*p))
      B_hat = IHT_greedy(Y_tilde, X_tilde, sparsity_lvls, num_iter=10)
      corr_emp = corr(B, B_hat)
      mean_final_corr_list_IHT[n_index] += corr_emp
      var_final_corr_list_IHT[run_num][n_index] = corr_emp

  mean_final_corr_list_AMP = mean_final_corr_list_AMP / num_runs
  SD_final_corr_list_AMP = np.sqrt(np.sum(np.square(var_final_corr_list_AMP - mean_final_corr_list_AMP), axis=0) / num_runs)
  print('mean_final_corr_list_AMP\n',mean_final_corr_list_AMP)
  print('SD_final_corr_list_AMP\n',SD_final_corr_list_AMP)

  mean_final_corr_list_NP = mean_final_corr_list_NP / num_runs
  SD_final_corr_list_NP = np.sqrt(np.sum(np.square(var_final_corr_list_NP - mean_final_corr_list_NP), axis=0) / num_runs)
  print('mean_final_corr_list_NP\n',mean_final_corr_list_NP)
  print('SD_final_corr_list_NP\n',SD_final_corr_list_NP)

  mean_final_corr_list_IHT = mean_final_corr_list_IHT / num_runs
  SD_final_corr_list_IHT = np.sqrt(np.sum(np.square(var_final_corr_list_IHT - mean_final_corr_list_IHT), axis=0) / num_runs)
  print('mean_final_corr_list_IHT\n',mean_final_corr_list_IHT)
  print('SD_final_corr_list_IHT\n',SD_final_corr_list_IHT)

  return [mean_final_corr_list_AMP, SD_final_corr_list_AMP, mean_final_corr_list_NP, SD_final_corr_list_NP,
          mean_final_corr_list_IHT, SD_final_corr_list_IHT]

def plot_AMP_v_NP_v_IHT_L3_noisy(alpha, p, B_bar_prob, sigma, n_list, num_iter, num_runs, num_MC_samples):

  B_bar_mean = B_bar_prob
  B_bar_cov = np.array([
      [B_bar_prob[0]-(B_bar_prob[0])**2, -1*B_bar_prob[0]*B_bar_prob[1], -1*B_bar_prob[0]*B_bar_prob[2]],
      [-1*B_bar_prob[1]*B_bar_prob[0], B_bar_prob[1]-(B_bar_prob[1])**2, -1*B_bar_prob[1]*B_bar_prob[2]],
      [-1*B_bar_prob[2]*B_bar_prob[0], -1*B_bar_prob[2]*B_bar_prob[1], B_bar_prob[2]-(B_bar_prob[2])**2]
      ])
  B_hat_0_row_mean = B_bar_mean
  B_hat_0_row_cov = B_bar_cov

  output_list = run_AMP_v_NP_v_IHT_L3_noisy(alpha, p, sigma, n_list, B_bar_prob, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, num_runs, num_MC_samples)

  mean_final_corr_list_AMP = output_list[0]
  SD_final_corr_list_AMP = output_list[1]
  mean_final_corr_list_NP = output_list[2]
  SD_final_corr_list_NP = output_list[3]
  mean_final_corr_list_IHT = output_list[4]
  SD_final_corr_list_IHT = output_list[5]

  # plotting beta1 sq norm correlation vs delta
  size = len(mean_final_corr_list_AMP)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_final_corr_list_AMP, yerr=SD_final_corr_list_AMP, color='blue', ecolor='blue', marker='o', elinewidth=3, capsize=10, label=r"AMP")
  plt.errorbar(delta_list, mean_final_corr_list_NP, yerr=SD_final_corr_list_NP, color='red', ecolor='red', marker='s', elinewidth=3, capsize=10, label=r"NP")
  plt.errorbar(delta_list, mean_final_corr_list_IHT, yerr=SD_final_corr_list_IHT, color='green', ecolor='green', marker='^', elinewidth=3, capsize=10, label=r"IHT")
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="upper left", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  #plt.savefig('.pdf', bbox_inches='tight')
  plt.show()

''' sigma = 0 '''

p = 500
alpha = 0.5
B_bar_prob = [1/3, 1/3, 1/3]
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
num_iter = 10
num_runs = 10
num_MC_samples = 500

B_bar_mean = B_bar_prob
B_bar_cov = np.array([
  [B_bar_prob[0]-(B_bar_prob[0])**2, -1*B_bar_prob[0]*B_bar_prob[1], -1*B_bar_prob[0]*B_bar_prob[2]],
  [-1*B_bar_prob[1]*B_bar_prob[0], B_bar_prob[1]-(B_bar_prob[1])**2, -1*B_bar_prob[1]*B_bar_prob[2]],
  [-1*B_bar_prob[2]*B_bar_prob[0], -1*B_bar_prob[2]*B_bar_prob[1], B_bar_prob[2]-(B_bar_prob[2])**2]
  ])
B_hat_0_row_mean = B_bar_mean
B_hat_0_row_cov = B_bar_cov

output_list = run_AMP_v_NP_v_IHT_L3_noiseless(alpha, p, n_list, B_bar_prob, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, num_runs, num_MC_samples)
save('GAMP_v_SE_3L_diff_algo_sig0', np.array(output_list))

''' sigma = 0.1 '''

p = 500
alpha = 0.5
B_bar_prob = [1/3, 1/3, 1/3]
sigma = 0.1
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
num_iter = 10
num_runs = 10
num_MC_samples = 500

B_bar_mean = B_bar_prob
B_bar_cov = np.array([
  [B_bar_prob[0]-(B_bar_prob[0])**2, -1*B_bar_prob[0]*B_bar_prob[1], -1*B_bar_prob[0]*B_bar_prob[2]],
  [-1*B_bar_prob[1]*B_bar_prob[0], B_bar_prob[1]-(B_bar_prob[1])**2, -1*B_bar_prob[1]*B_bar_prob[2]],
  [-1*B_bar_prob[2]*B_bar_prob[0], -1*B_bar_prob[2]*B_bar_prob[1], B_bar_prob[2]-(B_bar_prob[2])**2]
  ])
B_hat_0_row_mean = B_bar_mean
B_hat_0_row_cov = B_bar_cov

output_list = run_AMP_v_NP_v_IHT_L3_noisy(alpha, p, sigma, n_list, B_bar_prob, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, num_runs, num_MC_samples)
save('GAMP_v_SE_3L_diff_algo_sig01', np.array(output_list))

''' sigma = 0.3 '''

p = 500
alpha = 0.5
B_bar_prob = [1/3, 1/3, 1/3]
sigma = 0.3
n_list = [int(0.1*p), int(0.2*p), int(0.3*p), int(0.4*p), int(0.5*p), int(0.6*p), int(0.7*p), int(0.8*p), int(0.9*p), int(1.0*p)]
num_iter = 10
num_runs = 10
num_MC_samples = 500

B_bar_mean = B_bar_prob
B_bar_cov = np.array([
  [B_bar_prob[0]-(B_bar_prob[0])**2, -1*B_bar_prob[0]*B_bar_prob[1], -1*B_bar_prob[0]*B_bar_prob[2]],
  [-1*B_bar_prob[1]*B_bar_prob[0], B_bar_prob[1]-(B_bar_prob[1])**2, -1*B_bar_prob[1]*B_bar_prob[2]],
  [-1*B_bar_prob[2]*B_bar_prob[0], -1*B_bar_prob[2]*B_bar_prob[1], B_bar_prob[2]-(B_bar_prob[2])**2]
  ])
B_hat_0_row_mean = B_bar_mean
B_hat_0_row_cov = B_bar_cov

output_list = run_AMP_v_NP_v_IHT_L3_noisy(alpha, p, sigma, n_list, B_bar_prob, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov, num_iter, num_runs, num_MC_samples)
save('GAMP_v_SE_3L_diff_algo_sig03', np.array(output_list))
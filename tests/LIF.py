# Imports
import math

import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere

#print(t_max, dt, tau, el, vr, vth, r, i_mean)

# Loop for 10 steps, variable 'step' takes values from 0 to 9
# for step in range(10):

  # Compute value of t
  # t = step * dt

  # Compute value of i at this time step
  # i = i_mean*(1+np.sin(t*(2*np.pi/0.01)))

  # Print value of i
  # print(f'{t:.3f}' ' ' f'{i:.4e}')
  # print(f'{t:.3f} {i:.4e}')

  # Option key + 3 to write a hash
  # Control key + click -> "Execute Selection in Python Console" to run specific lines of code

  #x = 3.14159265e-1
  #print(f'{x:.3f}')
  #--> 0.314

  #print(f'{x:.4e}')
  #--> 3.1416e-01

  #################################################
  ## TODO for students: fill out compute v code ##
  # Fill out code and comment or remove the next line
  # raise NotImplementedError("Student exercise: You need to fill out code to compute v")
  #################################################

# Initialize step_end and v0
step_end = 10
v = el

  # Loop for step_end steps
for step in range(step_end):
  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Compute v
  v = v + (dt/tau)*(el - v + (r * i))

  # Print value of t and v
  print(f"{t:.3f} {v:.4e}")

#################################################
## TODO for students: fill out the figure initialization and plotting code below ##
# Fill out code and comment or remove the next line
#raise NotImplementedError("Student exercise: You need to fill out current figure code")
#################################################

# Initialize step_end
step_end = 25

# Initialize the figure
plt.figure()
plt.title("Change in Current per Unit of Time")
plt.xlabel("Time")
plt.ylabel("Current")

# Loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Plot i (use 'ko' to get small black dots (short for color='k' and marker = 'o'))
  plt.plot(t, i, 'ko')

# Display the plot
plt.show()

#################################################
## TODO for students: fill out the figure initialization and plotting code below ##
# Fill out code and comment or remove the next line
#raise NotImplementedError("Student exercise: You need to fill out membrane potential figure code")
#################################################

# Initialize step_end
step_end = int(t_max / dt)

# Initialize v0
v = el

# Initialize the figure
plt.figure()
plt.title('$V_m$ with sinusoidal I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)');

# Loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Compute v
  v = v + dt/tau * (el - v + r*i)

  # Plot v (using 'k.' to get even smaller markers)
  plt.plot(t, v, 'k.')

# Display plot
plt.show()

#################################################
## TODO for students: fill out code to get random input ##
# Fill out code and comment or remove the next line
#raise NotImplementedError("Student exercise: You need to fill out random input code")
#################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end and v
step_end = int(t_max / dt)
v = el

# Initialize the figure
plt.figure()
plt.title('$V_m$ with random I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

# loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Get random number in correct range of -1 to 1 (will need to adjust output of np.random.random)
  random_num = 2 * np.random.random() - 1

  # Compute value of i at this time step
  i = i_mean * (1 + 0.1 * math.sqrt(t_max/dt) * random_num)

  # Compute v
  v = v + dt/tau * (el - v + r*i)

  # Plot v (using 'k.' to get even smaller markers)
  plt.plot(t, v, 'k.')


# Display plot
plt.show()

# #################################################
# ## TODO for students: fill out code to store v in list ##
# # Fill out code and comment or remove the next line
# # raise NotImplementedError("Student exercise: You need to store v in list")
# #################################################
#
# # Set random number generator
# np.random.seed(2020)
#
# # Initialize step_end and n
# step_end = int(t_max / dt)
# n = 50
#
# # Intiatialize the list v_n with 50 values of membrane leak potential el
# v_n = [el] * 50
#
# #with plt.xkcd():
#   # Initialize the figure
# plt.figure()
# plt.title('Multiple realizations of $V_m$')
# plt.xlabel('time (s)')
# plt.ylabel('$V_m$ (V)')
#
#   # Loop for step_end steps
# for step in range(step_end):
#
#     # Compute value of t
#     t = step * dt
#
#     # Loop for n simulations
#     for j in range(0, n):
#
#       # Compute value of i at this time step
#       i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2* np.random.random() - 1))
#
#       # Compute value of v for this simulation
#       v_n[j] = v_n[j] + dt/tau * (el - v_n[j] + r*i)
#
#       # Compute sample mean by summing list of v_n using sum, and dividing by n
#       v_mean = sum(v_n)/n
#
#     # Plot all simulations (use alpha = 0.1 to make each marker slightly transparent)
#     plt.plot([t] * n, v_n, 'k.', alpha = 0.1)
#
#     #Plot sample mean using alpha=0.8 and 'C0.' for blue
#     plt.plot(t, v_mean, 'C0.', alpha = 0.8, markersize = 10)
#
#   # Display plot
# plt.show()

#################################################
## TODO for students: fill out code to plot sample standard deviation ##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to plot the sample standard deviation")
#################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end and n
step_end = int(t_max / dt)
n = 50

# Intiatialize the list v_n with 50 values of membrane leak potential el
v_n = [el] * n

# Initialize the figure
plt.figure()
plt.title('Multiple realizations of $V_m$')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

# Loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Loop for n simulations
  for j in range(0, n):

    # Compute value of i at this time step
    i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2* np.random.random() - 1))

    # Compute value of v for this simulation
    v_n[j] = v_n[j] + (dt / tau) * (el - v_n[j] + r*i)

  # Compute sample mean
  v_mean = sum(v_n) / n

  # Initialize a list `v_var_n` with the contribution of each V_n(t) to
  # Var(t) with a list comprehension over values of v_n
  v_var_n = [(v - v_mean)**2 for v in v_n]

  # Compute sample variance v_var by summing the values of v_var_n with sum and dividing by n-1
  v_var = sum(v_var_n)/(n-1)

  # Compute the standard deviation v_std with the function np.sqrt
  v_std = np.sqrt(v_var)

  # Plot simulations
  plt.plot(n*[t], v_n, 'k.', alpha=0.1)

  # Plot sample mean using alpha=0.8 and'C0.' for blue
  plt.plot(t, v_mean, 'C0.', alpha=0.8, markersize=10)

  # Plot mean + standard deviation with alpha=0.8 and argument 'C7'
  plt.plot(t, v_mean + v_std, 'C7.', alpha=0.8)

  # Plot mean - standard deviation with alpha=0.8 and argument 'C7'
  plt.plot(t, v_mean - v_std, 'C7.', alpha=0.8)

# Display plot
plt.show()


# Set random number generator
np.random.seed(2020)

# Initialize step_end and n
step_end = int(t_max / dt)
n = 50

# Intiatialize the list v_n with 50 values of membrane leak potential el
v_n = [el] * n

#with plt.xkcd():
  # Initialize the figure
plt.figure()
plt.title('Multiple realizations of $V_m$')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

  # Loop for step_end steps
for step in range(step_end):

    # Compute value of t
    t = step * dt

    # Loop for n simulations
    for j in range(0, n):

      # Compute value of i at this time step
      i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2* np.random.random() - 1))

      # Compute value of v for this simulation
      v_n[j] = v_n[j] + (dt / tau) * (el - v_n[j] + r*i)

    # Compute sample mean
    v_mean = sum(v_n) / n

    # Initialize a list `v_var_n` with the contribution of each V_n(t) to
    # Var(t) with a list comprehension over values of v_n
    v_var_n = [(v - v_mean)**2 for v in v_n]

    # Compute sample variance v_var by summing the values of v_var_n with sum and dividing by n-1
    v_var = sum(v_var_n) / (n - 1)

    # Compute the standard deviation v_std with the function np.sqrt
    v_std = np.sqrt(v_var)

    # Plot simulations
    plt.plot(n*[t], v_n, 'k.', alpha=0.1)

    # Plot sample mean using alpha=0.8 and'C0.' for blue
    plt.plot(t, v_mean, 'C0.', alpha=0.8, markersize=10)

    # Plot mean + standard deviation with alpha=0.8 and argument 'C7'
    plt.plot(t, v_mean + v_std, 'C7.', alpha=0.8)

    # Plot mean - standard deviation with alpha=0.8 and argument 'C7'
    plt.plot(t, v_mean - v_std, 'C7.', alpha=0.8)


  # Display plot
plt.show()



#################################################
## TODO for students: fill out code to rewrite simulation in numpy##
# Fill out code and comment or remove the next line
#raise NotImplementedError("Student exercise: You need to rewrite simulation in numpy")
#################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, v
step_end = int(t_max / dt) - 1
t_range = np.linspace(0, t_max, num=step_end, endpoint=False)
v = el * np.ones(step_end)
random_num = (2 * np.random.random(step_end) - 1)


# Simulate current over time

i = i_mean * (1 + 0.1 * (t_max/dt) ** (0.5) * random_num)
#i = i_mean * (1 + 0.1 * (t_max/dt) ** (0.5) * (2 * np.random.random(step_end) - 1))

# Loop for step_end steps
for step in range(1, step_end):

    # Compute v as function of i
    v[step] = v[step - 1] + (dt/tau) * (el - v[step - 1] + r * i[step - 1])

# Plot membrane potential
plt.figure()
plt.title('$V_m$ with random I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

plt.plot(t_range, v, 'k.')
plt.show()


#################################################
## TODO for students: fill out code to rewrite simulation in numpy##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to rewrite simulation in numpy")
#################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, v
step_end = int(t_max / dt) - 1
t_range = np.linspace(0, t_max, num=step_end, endpoint=False)
v = el * np.ones(step_end)

# Simulate current over time
i = i_mean * (1 + 0.1 * (t_max/dt) ** (0.5) * (2 * np.random.random(step_end) - 1))

# Loop for step_end values of i using enumerate
for step, i_step in enumerate(i):

  # Skip first iteration
  if step==0:
    continue

  # Compute v as function of i using i_step
  v[step] = v[step - 1] + (dt / tau) * (el - v[step - 1] + r * i_step)

# Plot figure
plt.figure()
plt.title('$V_m$ with random I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

plt.plot(t_range, v, 'k')
plt.show()


#################################################
## TODO for students: fill out code to use 2d arrays ##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to rewrite code to use 2d arrays")
#################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end, n, t_range, v and i
step_end = int(t_max / dt)
n = 50
t_range = np.linspace(0, t_max, num=step_end)
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max/dt) ** (0.5) * (2 * np.random.random([n,step_end]) - 1))

# Loop for step_end - 1 steps
for step in range(1, step_end):

   # Compute v_n
   v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

   # Compute sample mean (use np.mean)

   v_mean = np.mean(v_n, axis=0)

   # Compute sample standard deviation

   v_std = np.std(v_n, axis=0)

# Plot figure
with plt.xkcd():
    plt.figure()
    plt.title('Multiple realizations of $V_m$')
    plt.xlabel('time (s)')
    plt.ylabel('$V_m$ (V)')

    plt.plot(t_range, v_n.T, 'k', alpha=0.3)

    plt.plot(t_range, v_n[-1], 'k', alpha = 0.3, label = 'V(t)')
    plt.plot(t_range, v_mean, 'C0', alpha = 0.8, label = 'mean')
    plt.plot(t_range, v_mean+v_std, 'C7', alpha=0.8)
    plt.plot(t_range, v_mean-v_std, 'C7', alpha=0.8, label='mean $\pm$ std')

    plt.legend()
    plt.show()



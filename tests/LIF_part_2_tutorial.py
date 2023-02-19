# Imports

import numpy as np
import matplotlib.pyplot as plt

# @title Figure settings
import ipywidgets as widgets       # interactive display

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('retina')
#
# matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

# plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/content-creation/main/nma.mplstyle")

# @title Helper functions

t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere


def plot_all(t_range, v, raster=None, spikes=None, spikes_mean=None):
  """
  Plots Time evolution for
  (1) multiple realizations of membrane potential
  (2) spikes
  (3) mean spike rate (optional)

  Args:
    t_range (numpy array of floats)
        range of time steps for the plots of shape (time steps)

    v (numpy array of floats)
        membrane potential values of shape (neurons, time steps)

    raster (numpy array of floats)
        spike raster of shape (neurons, time steps)

    spikes (dictionary of lists)
        list with spike times indexed by neuron number

    spikes_mean (numpy array of floats)
        Mean spike rate for spikes as dictionary

  Returns:
    Nothing.
  """

  v_mean = np.mean(v, axis=0)
  fig_w, fig_h = plt.rcParams['figure.figsize']
  plt.figure(figsize=(fig_w, 1.5 * fig_h))

  ax1 = plt.subplot(3, 1, 1)
  for j in range(n):
    plt.scatter(t_range, v[j], color="k", marker=".", alpha=0.01)
  plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
  plt.xticks([])
  plt.ylabel(r'$V_m$ (V)')

  if raster is not None:
    plt.subplot(3, 1, 2)
    spikes_mean = np.mean(raster, axis=0)
    plt.imshow(raster, cmap='Greys', origin='lower', aspect='auto')

  else:
    plt.subplot(3, 1, 2, sharex=ax1)
    for j in range(n):
      times = np.array(spikes[j])
      plt.scatter(times, j * np.ones_like(times), color="C0", marker=".", alpha=0.2)

  plt.xticks([])
  plt.ylabel('neuron')

  if spikes_mean is not None:
    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(t_range, spikes_mean)
    plt.xlabel('time (s)')
    plt.ylabel('rate (Hz)')

  plt.tight_layout()
  plt.show()
  

  # plt.hist(data, bins, label='my data')
  # plt.legend()
  # plt.show()


#################################################
## TODO for students: fill out code to plot histogram ##
# Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to plot histogram")
#################################################

# # Set random number generator
# np.random.seed(2020)
#
# # Initialize t_range, step_end, n, v_n, i and nbins
# t_range = np.arange(0, t_max, dt)
# step_end = len(t_range)
# n = 10000
# v_n = el * np.ones([n, step_end])
# i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))
# nbins = 32
#
# # Loop over time steps
# for step, t in enumerate(t_range):
#
#   # Skip first iteration
#   if step==0:
#     continue
#
#   # Compute v_n
#   v_n[:, step] =  v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])
#
# # Initialize the figure
# plt.figure()
# plt.ylabel('Frequency')
# plt.xlabel('$V_m$ (V)')
#
# # Plot a histogram at t_max/10 (add labels and parameters histtype='stepfilled' and linewidth=0)
# plt.hist(v_n[:,int(step_end / 10)], nbins,
#          histtype='stepfilled', linewidth=0,
#          label = 't='+ str(t_max / 10) + 's')
#
# # Plot a histogram at t_max (add labels and parameters histtype='stepfilled' and linewidth=0)
# plt.hist(v_n[:,-1], nbins,
#          histtype='stepfilled', linewidth=0,
#          label = 't='+ str(t_max) + 's')
#
# # Add legend
# plt.legend()
# plt.show()

# mydict = {1: 'Lihua', 'grade':100}
# print(mydict.keys())
# print(mydict.values())
# a = list(mydict.keys())
# print(a)
#
# for item in mydict:
#   print(item, mydict[item])
#
#
# mydict = {x: x**2 for x in range(1,4)}
# for item in mydict:
#   print(item, mydict[item])
#
# mydict = {9: 'Lihua', 1: 'Aisha'}
#
# for item in mydict:
#   print(mydict[item])



# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

#################################################
## TODO for students: add spikes to LIF neuron ##
# Fill out function and remove
# raise NotImplementedError("Student exercise: add spikes to LIF neuron")
#################################################

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step==0:
    continue

  # Compute v_n
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])

  # Loop over simulations
  for j in range(n):

    # Check if voltage above threshold
    if v_n[j, step] >= vth:

      # Reset to reset voltage
      v_n[j, step] = vr

      # Add this spike time
      spikes[j] += [t]

      # Add spike count to this step
      spikes_n[step] += 1

# Collect mean Vm and mean spiking rate
v_mean = np.mean(v_n, axis=0)
spikes_mean =  spikes_n / n

# Initialize the figure
plt.figure()

# Plot simulations and sample mean
ax1 = plt.subplot(3, 1, 1)
for j in range(n):
  plt.scatter(t_range, v_n[j], color="k", marker=".", alpha=0.01)
plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
plt.ylabel('$V_m$ (V)')

# Plot spikes
plt.subplot(3, 1, 2, sharex=ax1)
# for each neuron j: collect spike times and plot them at height j
for j in range(n):
  times = np.array(spikes[j])
  plt.scatter(times, j + np.ones_like(times), color="C0", marker=".", alpha=0.2)

plt.ylabel('neuron')

# Plot firing rate
plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(t_range, spikes_mean)
plt.xlabel('time (s)')
plt.ylabel('rate (Hz)')

plt.tight_layout()
plt.show()

# #for some reason code below does not generate a figure - I should have added plt.show() in order to show the figure!
#
#
# # Set random number generator
# np.random.seed(2020)
#
# # Initialize step_end, t_range, n, v_n and i
# t_range = np.arange(0, t_max, dt)
# step_end = len(t_range)
# n = 500
# v_n = el * np.ones([n, step_end])
# i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))
#
# # Initialize spikes and spikes_n
# spikes = {j: [] for j in range(n)}
# spikes_n = np.zeros([step_end])
#
# # Loop over time steps
# for step, t in enumerate(t_range):
#
#   # Skip first iteration
#   if step==0:
#     continue
#
#   # Compute v_n
#   v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])
#
#   # Loop over simulations
#   for j in range(n):
#
#     # Check if voltage above threshold
#     if v_n[j, step] >= vth:
#
#       # Reset to reset voltage
#       v_n[j, step] = vr
#
#       # Add this spike time
#       spikes[j] += [t]
#
#       # Add spike count to this step
#       spikes_n[step] += 1
#
# # Collect mean Vm and mean spiking rate
# v_mean = np.mean(v_n, axis=0)
# spikes_mean =  spikes_n / n
#
#
# # Initialize the figure
# plt.figure()
#
# # Plot simulations and sample mean
# ax1 = plt.subplot(3, 1, 1)
# for j in range(n):
#   plt.scatter(t_range, v_n[j], color="k", marker=".", alpha=0.01)
# plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
# plt.ylabel('$V_m$ (V)')
#
# # Plot spikes
# plt.subplot(3, 1, 2, sharex=ax1)
# # for each neuron j: collect spike times and plot them at height j
# for j in range(n):
#   times = np.array(spikes[j])
#   plt.scatter(times, j * np.ones_like(times), color="C0", marker=".", alpha=0.2)
#
# plt.ylabel('neuron')
#
# # Plot firing rate
# plt.subplot(3, 1, 3, sharex=ax1)
# plt.plot(t_range, spikes_mean)
# plt.xlabel('time (s)')
# plt.ylabel('rate (Hz)')
#
# plt.tight_layout()
#
# plt.show()

# @markdown Execute this cell to enable the demo

def random_ref_period(mu, sigma):
  # set random number generator
  np.random.seed(2020)

  # initialize step_end, t_range, n, v_n, syn and raster
  t_range = np.arange(0, t_max, dt)
  step_end = len(t_range)
  n = 500
  v_n = el * np.ones([n,step_end])
  syn = i_mean * (1 + 0.1*(t_max/dt)**(0.5)*(2*np.random.random([n,step_end])-1))
  raster = np.zeros([n,step_end])

  # initialize t_ref and last_spike
  t_ref = mu + sigma*np.random.normal(size=n)
  t_ref[t_ref<0] = 0
  last_spike = -t_ref * np.ones([n])

  # loop time steps
  for step, t in enumerate(t_range):
    if step==0:
      continue

    v_n[:,step] = v_n[:,step-1] + dt/tau * (el - v_n[:,step-1] + r*syn[:,step])

    # boolean array spiked indexes neurons with v>=vth
    spiked = (v_n[:,step] >= vth)
    v_n[spiked,step] = vr
    raster[spiked,step] = 1.

    # boolean array clamped indexes refractory neurons
    clamped = (last_spike + t_ref > t)
    v_n[clamped,step] = vr
    last_spike[spiked] = t

  # plot multiple realizations of Vm, spikes and mean spike rate
  plot_all(t_range, v_n, raster)

  # plot histogram of t_ref
  plt.figure(figsize=(8,4))
  plt.hist(t_ref, bins=32, histtype='stepfilled', linewidth=0, color='C1')
  plt.xlabel(r'$t_{ref}$ (s)')
  plt.ylabel('count')
  plt.tight_layout()

_ = widgets.interact(random_ref_period, mu = (0.01, 0.05, 0.01), \
                              sigma = (0.001, 0.02, 0.001))

def ode_step(v, i, dt):
  """
  Evolves membrane potential by one step of discrete time integration

  Args:
    v (numpy array of floats)
      membrane potential at previous time step of shape (neurons)

    v (numpy array of floats)
      synaptic input at current time step of shape (neurons)

    dt (float)
      time step increment

  Returns:
    v (numpy array of floats)
      membrane potential at current time step of shape (neurons)
  """
  v = v + dt/tau * (el - v + r*i)

  return v

def spike_clamp(v, delta_spike):
  """
  Resets membrane potential of neurons if v>= vth
  and clamps to vr if interval of time since last spike < t_ref

  Args:
    v (numpy array of floats)
      membrane potential of shape (neurons)

    delta_spike (numpy array of floats)
      interval of time since last spike of shape (neurons)

  Returns:
    v (numpy array of floats)
      membrane potential of shape (neurons)
    spiked (numpy array of floats)
      boolean array of neurons that spiked  of shape (neurons)
  """

  ####################################################
  ## TODO for students: complete spike_clamp
  # Fill out function and remove
  #raise NotImplementedError("Student exercise: complete spike_clamp")
  ####################################################
  # Boolean array spiked indexes neurons with v>=vth
  spiked = (v >= vth)
  v[spiked] = vr

  # Boolean array clamped indexes refractory neurons
  clamped = (t_ref > delta_spike)
  v[clamped] = vr

  return v, spiked


# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Initialize t_ref and last_spike
mu = 0.01
sigma = 0.007
t_ref = mu + sigma*np.random.normal(size=n)
t_ref[t_ref<0] = 0
last_spike = -t_ref * np.ones([n])

# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step==0:
    continue

  # Compute v_n
  v_n[:,step] = ode_step(v_n[:,step-1], i[:,step], dt)

  # Reset membrane potential and clamp
  v_n[:,step], spiked = spike_clamp(v_n[:,step], t - last_spike)

  # Update raster and last_spike
  raster[spiked,step] = 1.
  last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)


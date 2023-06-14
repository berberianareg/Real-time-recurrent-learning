"""Real-time recurrent learning (RTRL) for continuous online supervised learning.

Notes
-----
  This script is version v0. It provides the base for all subsequent
  iterations of the project.

Requirements
------------
  See "requirements.txt"
  
Notes
-----
  Fully connected recurrent network with 2 inputs, 3 hidden units, and 1 output unit.
  Activation function takes the form of a logistic function.

"""

#%% import libraries and modules
import os
import numpy as np  
import matplotlib.pyplot as plt

#%% figure parameters
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['font.size']= 15
plt.rcParams['lines.linewidth'] = 4

#%%

class RTRL:
    """Real time recurrent learning class."""
    
    def __init__(self, dim_input=2, dim_hidden=3, dim_output=1, min_initial_w=0.01, max_initial_w=0.1, learning_rate=10, num_timesteps=100):
        # dimensionality of input space consisting of source nodes (default: 2)
        self.dim_input = dim_input
        # dimensionality of state space consisting of hidden units (default: 3)
        self.dim_hidden = dim_hidden
        # dimensionality of output space consisting of output units (default: 1)
        self.dim_output = dim_output
        # min initial connection weight (default: 0.01)
        self.min_initial_w = min_initial_w
        # max initial connection weight (default: 0.1)
        self.max_initial_w = max_initial_w
        # learning rate (default: 10)
        self.learning_rate = learning_rate
        # number of timesteps (default: 100)
        self.num_timesteps = num_timesteps
    
    def activation_function(self, y):
        """Apply logistic function as activation function."""
        f_y = 1 / (1 + np.exp(-y))
        
        return f_y
    
    def initialize_network(self):
        """Initialize network state and partial derivative of network state."""
        # connection weights of units in the hidden layer connected to feedback nodes in the input layer
        w_hidden = np.random.uniform(low=self.min_initial_w, high=self.max_initial_w, size=(self.dim_hidden, self.dim_hidden))
        
        # connection weights of units in the hidden layer connected to source nodes in the input layer
        w_input = np.random.uniform(low=self.min_initial_w, high=self.max_initial_w, size=(self.dim_hidden, self.dim_input))
        # bias terms applied to hidden units
        w_input = np.hstack((np.ones([self.dim_hidden, 1]), w_input))
        
        # connection weights of units in the output layer connected to hidden units
        w_output = np.eye(self.dim_output, self.dim_hidden)
        
        # initialize network state
        state = np.zeros([self.dim_hidden, 1])
        
        # initialize partial derivative of the network state with respect to the weights
        state_change = np.zeros([self.dim_hidden, self.dim_hidden, self.dim_hidden + self.dim_input + 1])
        
        return w_hidden, w_input, w_output, state, state_change
    
    def run_model(self):
        """Train model."""
        # empty lists for variable storage
        input_list = []
        desired_output_list = []
        actual_output_list = []
        error_list = []
        
        # initialize network
        w_hidden, w_input, w_output, state, state_change = self.initialize_network()
        
        # concatenate feedback and feedforward connection weights
        w = np.hstack((w_hidden, w_input)).T
        
        for timestep in range(self.num_timesteps):
            # specify input
            inp = np.zeros([self.dim_input, 1]) + 0.5
            input_list.append(inp)
            
            # specify desired output
            desired_output = np.zeros([self.dim_output, 1]) + 0.5
            desired_output_list.append(desired_output)
            
            # compute actual output
            actual_output = np.dot(w_output, state)
            actual_output_list.append(actual_output)
            
            # compute error
            error = desired_output - actual_output
            error_list.append(0.5 * np.dot(error.T, error).squeeze())
            
            # compute change in connection weights
            delta_w = self.learning_rate * np.dot(np.dot(w_output, state_change).T, error).squeeze()
            
            # concatenate state, [bias], input
            z = np.vstack((state, [1], inp))
            
            # compute next state
            state = self.activation_function(np.dot(w.T, z))
            
            # extract feedback connection weights
            w_hidden = w[:self.dim_hidden, :self.dim_hidden]
            
            # compute partial derivative of state with respect to connection weights
            state_change = state * (1 - state) * (np.dot(w_hidden, state_change) + z.T)
            
            # update connection weights
            w = w + delta_w
            
        return input_list, desired_output_list, actual_output_list, error_list
    
    def plot_activation_function(self):
        """Plot activation function."""
        y = np.linspace(-5, 5, 1000)
        fig, ax = plt.subplots()
        plt.plot(y, self.activation_function(y), color='k')
        plt.title('Logistic function')
        plt.xlabel('y')
        plt.ylabel('f(y)')
        plt.tight_layout()
        fig.savefig(os.path.join(os.getcwd(), 'figure_1'))

    def plot_model_results(self, output_idx=0):
        """Plot model results."""
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
        ax1.plot(range(model.num_timesteps), np.array(input_list)[:, output_idx, :], color='r')
        ax1.set_title('Input')
        ax1.set_ylim(0, 1)

        ax2.plot(range(model.num_timesteps), np.array(desired_output_list)[:, output_idx, :], label='desired', color='b')
        ax2.plot(range(model.num_timesteps), np.array(actual_output_list)[:, output_idx, :], label='actual', color='k')
        ax2.set_title('Output')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right', frameon=False, ncol=2)

        ax3.plot(range(model.num_timesteps), error_list, color='k')
        ax3.set_title('Error')

        plt.xlabel('Timestep')
        plt.tight_layout()
        fig.savefig(os.path.join(os.getcwd(), 'figure_2'))


#%% instantiate RTRL class
model = RTRL()

#%% run model
input_list, desired_output_list, actual_output_list, error_list = model.run_model()

#%% plot figures

cwd = os.getcwd()                                                               # get current working directory
fileName = 'images'                                                             # specify filename

# filepath and directory specifications
if os.path.exists(os.path.join(cwd, fileName)) == False:                        # if path does not exist
    os.makedirs(fileName)                                                       # create directory with specified filename
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory
else:
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory

model.plot_activation_function()
model.plot_model_results()

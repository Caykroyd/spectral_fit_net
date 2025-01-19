import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections.abc import Mapping, Sequence

class Parameter:
    def __init__(self, key : tuple, lims : tuple):
        self.key = key
        self.lims = lims

    def __repr__(self):
        return f'Parameter(key={self.key}, lims={self.lims})'

    def normalise(self, p):
        a, b = self.lims
        assert a < b, f'Invalid limits for {self.key}: lower bound must be less than upper bound ({self.lims})'
        return (p - a) / (b - a)
    
    def denormalise(self, x):
        a, b = self.lims
        assert a < b, f'Invalid limits for {self.key}: lower bound must be less than upper bound ({self.lims})'
        return a + x * (b - a)

class Pattern:
    @classmethod
    def match(cls, key, pattern, wildcard=None):
        if key == pattern:
            return True
        elif isinstance(pattern, tuple):
            return Pattern._match_tuple(key, pattern, wildcard)
        else:
            return False

    @classmethod
    def _match_tuple(cls, key, pattern, wildcard=None):
        if len(pattern) != len(key):
                return False
        return all(
            (p is wildcard) or (k is wildcard) or (p == k)
            for p, k in zip(pattern, key)
        )
    
    @classmethod
    def select(cls, pattern, key_list, wildcard=None):
        return filter(lambda p : Pattern.match(p, pattern, wildcard), key_list)

class SpectralWindow:
    def __init__(self, name : str, emlines : list, x_lims : tuple = None, wl = None, **kwargs):
        '''
        Represents a spectral window.

        Args:
            name (str): Name of the 
            ...
            kwargs: additional metadata to be saved on this instance
        '''
        self.name = name
        self.x_lims = x_lims
        self.lines = emlines
        self.wl = wl
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self):
        return f'SpectralWindow(name={self.name}, x_lims={self.x_lims}, lines={self.lines})'


class EmissionLine:
    def __init__(self, name : str, channel : str, num_components : int, constraints : dict = None, par_lims = None, **kwargs):
        '''
        Represents an emission line with potential parameter constraints.

        Args:
            name (str): Name of the emission line.
            channel: channel where this line is present.
            num_components (int): Number of Gaussian components for this line.
            constraints: Dictionary of constraints on parameters. Each constraint is a lambda function depending on free parameters.
            kwargs: additional metadata to be saved on this instance
        '''
        self.name = name
        self.channel = channel
        self.num_components = num_components
        self.constraints = constraints or {}
        # Save metadata
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.build_parameters(par_lims)

    def __repr__(self):
        return f'EmissionLine(name={self.name}, channel={self.channel}, ncomp={self.num_components})'

    def build_parameters(self, par_lims = None):
        
        def par_lims_func(par):
            '''Returns parameter limits based on input type.'''
            if par_lims is None:
                lims = (0, 1)
            elif isinstance(par_lims, Sequence) and not isinstance(par_lims, (str, Mapping)):
                lims = par_lims
            elif isinstance(par_lims, Mapping):
                lims = par_lims.get(par, [0, 1])  
            else:
                raise ValueError(
                    'Parameter limits must be either:\n'
                    '- A tuple (min, max)\n'
                    '- A dictionary with keys {"amp", "mu", "sigma"} and tuple values'
                )
            
            # Ensure it's exactly (min, max)
            if len(lims) != 2:
                    raise ValueError('Parameter limits sequence must have exactly two elements (min, max).')
            
            return tuple(lims)
            
        # create parameters for each Gaussian component
        # TODO: If parameter is not a tied parameter, keep lims = None
        self.parameters = [
            Parameter(
                key=(self.name, param_name, comp_idx), 
                lims=par_lims_func(param_name)
            ) 
            for param_name in ['amp', 'mu', 'sigma'] 
            for comp_idx in range(self.num_components) 
        ]

class ParameterMapping:
    def __init__(self, parameters : list = None, constraints : dict = None):
        self.parameters = parameters or []
        self.prepare_constraints(constraints)

    # @property
    # def tied_params(self):
    #     return self.constraints.keys()

    # @property
    # def free_params(self):
    #     return (p.key for p in self.parameters if p.key not in self.constraints)

    def prepare_constraints(self, constraints):

        # Validate constraints
        constraints = constraints or {}

        for key, fun in constraints.items():
            if not callable(fun):
                raise ValueError(f'Invalid constraint function for parameter {key}: {fun}.')

        # Build parameter lists
        self.free_params = []
        self.tied_params = []
        self.constraints = {} # we replace wildcards with their possible values 

        for param in self.parameters:
            key = param.key

            # Select constraints which match with parameter key
            matches = list(filter(lambda patt : Pattern.match(key, patt, wildcard=None), constraints))

            if len(matches) > 1:
                raise ValueError(f'Multiple patterns match key={key}: {matches}')
            
            elif matches:
                self.tied_params.append(param)
                self.constraints[key] = constraints[matches[-1]] # set constraint to matched value
            else:
                self.free_params.append(param)
                
    def unpack_tensor(self, network_outputs):
        '''
        Maps network outputs to parameter dictionary, denormalizing and applying constraints.

        Args:
            network_outputs (Tensor): Output tensor from the neural network of shape (batch_size, free_param_count).

        Returns:
            params (dict): Dictionary of parameters with keys (line_name, param_name, component) and value (Tensor) of shape (batch_size,)
        '''
        assert network_outputs.shape[1] == len(self.free_params), f'Parameter count ({len(self.free_params)}) is incompatible with network output size ({network_outputs.shape[1]}) !'

        # 1. Assign free params to output dict
        # Values must be de-normalised
        param_dict = {par.key: par.denormalise(network_outputs[..., idx])
                  for idx, par in enumerate(self.free_params)}
        
        # 2. Assign non-free params to output dict
        # Constraints are assumed to apply to unnormalised values
        for param in self.tied_params:
            key = param.key
            fun = self.constraints[key] # find the constraint function which matches key
            param_dict[key] = fun(param_dict, *key) # can throw an error if parameter in fun hasn't been defined
        
        # for key, fun in self.constraints.items():
        #     param_dict[key] = fun(param_dict, *key) # can throw an error if parameter in fun hasn't been defined
        
        return param_dict

    def pack_tensor(self, param_dict):
        '''
        Packs free parameters into a normalized tensor ready for the neural network.
        Args:
            param_dict (dict): Dictionary where keys are parameter names matching `self.free_params`
                            and values are tensors or arrays of shape `(batch_size,)`.

        Returns:
            param_tensor (Tensor): Tensor of shape `(batch_size, num_free_params)` containing the packed parameters.
        '''
        # Set of keys of all free parameters
        free_keys = {param.key for param in self.free_params}

        # Ensure that all required parameters are present
        missing_keys = free_keys - param_dict.keys()
        if missing_keys:
            raise KeyError(f'The following parameters are missing in param_dict: {missing_keys}')

        # Check for extra keys
        extra_keys = param_dict.keys() - free_keys
        if extra_keys:
            print(f'Warning: param_dict contains extra parameters that will be ignored: {extra_keys}')

        # Normalise parameters and stack them together
        return torch.stack(
                [param.normalise(param_dict[param.key]) for param in self.free_params]
            , dim = -1)
    
class GaussianSuperposition(nn.Module):
    def __init__(self, emission_lines, channels, parameter_mapping):
        super(GaussianSuperposition, self).__init__()
        self.mapping = parameter_mapping
        self.emission_lines = emission_lines
        self.channels = channels
        self.channels_index_map = {key: i for i, key in enumerate(channels)}

    @property
    def max_lines_per_channel(self):
         # There may be a variable amount of lines in each channel
        return max(len(ch.lines) for ch in self.channels.values())

    @property 
    def max_components_per_line(self):
         # There may be a variable amount of components in each line
        return max(line.num_components for line in self.emission_lines.values())

    @property
    def signal_lims(self):
        return [ch for ch in self.channels.values()]

    def components(self, x, network_outputs):
        '''
        Computes the separate Gaussians components for each channel.

        Args:
            x (Tensor): Input wavelengths of shape (batch_size, channels, data_length).
            network_outputs (Tensor): Output tensor from the neural network of shape (batch_size, free_param_count).

        Returns:
            output (Tensor): Gaussians of shape (batch_size, channels, lines_per_channel, components_per_line, data_length).

        '''
        LpC = self.max_lines_per_channel
        KpL = self.max_components_per_line

        # Get batch size (N), channel count (C), and data length (D)
        N, C, D = x.shape

        # Get batch size (N), free_params_count (P)
        N, P = network_outputs.shape

        # Extract parameters from the network output
        params = self.mapping.unpack_tensor(network_outputs)
        
        # Initialize parameter tensors, which can be relatively sparse
        # Elements which aren't filled keep amp = 0 and so return an empty signal
        # Because torch does not allow assignment to slices when grad is required, we need to create lists of lists and stack...
        amp = torch.zeros(N, C, LpC, KpL, device=x.device)
        mu  = torch.zeros(N, C, LpC, KpL, device=x.device)
        sigma = torch.ones(N, C, LpC, KpL, device=x.device)

        # Fill parameter tensors with values obtained from the mapping
        for ch_idx, ch in enumerate(self.channels.values()):
            for ln_idx, ln_name in enumerate(ch.lines):
                ln = self.emission_lines[ln_name]
                ncomp = ln.num_components
                for comp_idx in range(ncomp):
                    amp[:, ch_idx, ln_idx, comp_idx]   = params[ln_name, 'amp', comp_idx]
                    mu[:, ch_idx, ln_idx, comp_idx]    = params[ln_name, 'mu', comp_idx]
                    sigma[:, ch_idx, ln_idx, comp_idx] = params[ln_name, 'sigma', comp_idx] 

        # for line in self.emission_lines:
        #     channel = self.channels[line.channel]
        #     channel_idx = self.channels_index_map[channel.name]
        #     line_idx = channel.lines.index(line.name)
        #     for comp_idx in range(line.num_components):
        #         amp[:, channel_idx, line_idx, comp_idx]   = params[line.name, 'amp', comp_idx]
        #         mu[:, channel_idx, line_idx, comp_idx]    = params[line.name, 'mu', comp_idx]
        #         sigma[:, channel_idx, line_idx, comp_idx] = params[line.name, 'sigma', comp_idx]
        
        # TODO: direct mapping of the free params with tensor indexing?? could be more efficient
        # index_map, index_mask = self.mapping.index_map_and_mask

        # Expand dimensions for broadcasting
        x = x[..., None, None, :]  # (N, C, 1, 1, D)

        # Compute each Gaussian
        y = amp[..., None] * torch.exp(- 0.5 * (x - mu[..., None])**2 / sigma[..., None]**2)
        
        return y

    def forward(self, x, network_outputs):
        '''
        Computes the superposition of Gaussians for each channel.

        Args:
            x (Tensor): Input wavelengths of shape (batch_size, channels, data_length).
            network_outputs (Tensor): Output tensor from the neural network of shape (batch_size, free_param_count).

        Returns:
            output (Tensor): Superposed Gaussians of shape (batch_size, channels, data_length).
        '''

        y = self.components(x, network_outputs) # shape (batch_size, channels, max_lines, max_components, data_length)

        return torch.sum(y, dim=(-2,-3)) # sum along components & spectral lines in channel


    def fit(self, x, params, z, lr=5e-2, nsteps=100, err='mse', reg=None, verbose=False):
        # fits parameters to data series z(x), using current parameters as initial guess
        
        params = nn.Parameter(params.clone().detach())
        # possible masking of parameters that are fixed

        optimizer = optim.Adam([params], lr=lr, amsgrad=True)
        
        if isinstance(err, str):
            err = dict.get({
                'mse': F.mse_loss
            }, err)
        
        if reg is None:
            reg = lambda par: 0

        for epoch in range(nsteps):
            optimizer.zero_grad()
            signal = self(x, params)
            loss = err(signal, z) + reg(params)
            if verbose: 
                print(f'Fit epoch {epoch+1}: loss = {loss.item():.3e}')#, f'  Amp: {amp}', f'  mu: {mu},', f'  sigma: {sigma}', sep='\n')
            loss.backward()
            optimizer.step()
        
        # Return fitted values
        return params.detach()

# # Example use

# # When emission lines are used, free parameters are automatically created for any unconstrained params found.
# lines = [
#     EmissionLine('Halpha', rest_wavelength=None, channel=0, num_components=2, constraints=None),
#     EmissionLine('Hbeta', rest_wavelength=None, channel=1, num_components=2, constraints={
#         'amp' : lambda params, ln, par, comp : 3*params['Halpha', 'amp', comp], # free parameter is defined automatically in previous line from all non-constrained params 
#     }),
# ]

# # Sometimes we wish to define certain free parameters explicitly. We need only define their names; these can be strings or tuples.
# free_params = [
#     'free-param',
#     *(('Hgamma', 'free-mu', i) for i in range(2)), # one for each component
# ]

# lines.append(
#     EmissionLine('Hgamma', rest_wavelength=None, channel=1, num_components=2, constraints={\
#         'amp' : lambda params, *_ : params['free-param'], # free parameter is defined manually
#         'mu'  : lambda params, ln, par, com : params['Hgamma', 'free-mu', com], # free parameter is defined manually
#     })
# )
# Idea to improve constraint setup: make params not a dict but a function. This way we can have special keywords for each argument if it means comp_idx, line.name, etc

# parameter_mapping = ParameterMapping(emission_lines=lines, free_parameters=free_params)

# # Simulate network outputs
# batch_size = 3
# free_param_count = len(parameter_mapping.free_params)
# network_outputs = torch.randn(batch_size, free_param_count, requires_grad=True)
# print(network_outputs)

# # Input wavelengths
# channels = 2  # Example number of channels
# data_length = 100  # Example data length
# x = torch.linspace(-1, 1, data_length).unsqueeze(0).unsqueeze(0).expand(batch_size, channels, data_length)

# # Create GaussianSuperposition instance
# gaussian_model = GaussianSuperposition(emission_lines=lines, parameter_mapping=parameter_mapping)

# # Forward pass
# output = gaussian_model(x, network_outputs)

# # Define target (e.g., zeros for simplicity)
# target = torch.zeros_like(output)

# # Compute loss
# loss = torch.nn.functional.mse_loss(output, target)

# # Backward pass
# loss.backward()
# print(loss.item())

# # Check gradients
# print('Gradients w.r.t network outputs:')
# print(network_outputs.grad)

# # Check if gradients are flowing
# if network_outputs.grad is not None and torch.any(network_outputs.grad != 0):
#     print('Gradients are flowing back to network outputs.')
# else:
#     print('No gradients flowing back to network outputs.')


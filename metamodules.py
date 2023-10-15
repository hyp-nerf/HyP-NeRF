import torch
import re
from torch import nn
from torchmeta.modules import MetaModule, MetaSequential
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)

class BatchLinear(nn.Linear, MetaModule):
	'''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
	hypernetwork.'''
	__doc__ = nn.Linear.__doc__

	def forward(self, input, params=None):
		if params is None:
			params = OrderedDict(self.named_parameters())
		if self.bias is not None:
			bias = params.get('bias', None)
		weight = params['weight']

		output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))

		if self.bias is not None:
			output += bias.unsqueeze(-2)

		return output


class Sine(nn.Module):
	def __init(self):
		super().__init__()

	def forward(self, input):
		# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
		return torch.sin(30 * input)

class FCBlock(MetaModule):
	'''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
	Can be used just as a normal neural network though, as well.
	'''

	def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
				 outermost_linear=False, nonlinearity='relu', weight_init=None,bias = True):
		super().__init__()

		# nonlinearity = 'sine'

		self.first_layer_init = None

		# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
		# special first-layer initialization scheme
		nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
						 'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
						 'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
						 'tanh':(nn.Tanh(), init_weights_xavier, None),
						 'selu':(nn.SELU(inplace=True), init_weights_selu, None),
						 'softplus':(nn.Softplus(), init_weights_normal, None),
						 'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

		nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

		if weight_init is not None:  # Overwrite weight init if passed
			self.weight_init = weight_init
		else:
			self.weight_init = nl_weight_init

		self.net = []
		self.net.append(MetaSequential(
			BatchLinear(in_features, hidden_features,bias=bias), nl
		))

		for i in range(num_hidden_layers):
			self.net.append(MetaSequential(
				BatchLinear(hidden_features, hidden_features,bias=bias), nl
			))

		if outermost_linear:
			self.net.append(MetaSequential(BatchLinear(hidden_features, out_features,bias=bias)))
		else:
			self.net.append(MetaSequential(
				BatchLinear(hidden_features, out_features,bias=bias), nl
			))

		self.net = MetaSequential(*self.net)
		if self.weight_init is not None:
			self.net.apply(self.weight_init)

		if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
			self.net[0].apply(first_layer_init)

	def forward(self, coords, params=None, **kwargs):
		if params is None:
			params = OrderedDict(self.named_parameters())

		# print('passing on with siren ', siren, get_subdict(params, 'net').keys())
		output = self.net(coords, params=get_subdict(params, 'net'))
		# output = self.net(coords)
		return output

	def forward_with_activations(self, coords, params=None, retain_grad=False):
		'''Returns not only model output, but also intermediate activations.'''
		if params is None:
			params = OrderedDict(self.named_parameters())

		activations = OrderedDict()

		x = coords.clone().detach().requires_grad_(True)
		activations['input'] = x
		for i, layer in enumerate(self.net):
			subdict = get_subdict(params, 'net.%d' % i)
			for j, sublayer in enumerate(layer):
				if isinstance(sublayer, BatchLinear):
					x = sublayer(x, params=get_subdict(subdict, '%d' % j))
				else:
					x = sublayer(x)

				if retain_grad:
					x.retain_grad()
				activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
		return activations

########################
# HyperNetwork modules
class HyperNetwork(nn.Module):
	def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module,activation='relu'):
		'''
		Args:
			hyper_in_features: In features of hypernetwork
			hyper_hidden_layers: Number of hidden layers in hypernetwork
			hyper_hidden_features: Number of hidden units in hypernetwork
			hypo_module: MetaModule. The module whose parameters are predicted.
		'''
		super().__init__()

		hypo_parameters = hypo_module.meta_named_parameters()

		self.names = []
		self.nets = nn.ModuleList()
		self.param_shapes = []

		for name, param in hypo_parameters:
			if 'variance' in name:
				continue
			self.names.append(name)
			self.param_shapes.append(param.size())

			hn = FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
					num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
					outermost_linear=True,nonlinearity=activation)

			if 'weight' in name:
				hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
			elif 'bias' in name:
				hn.net[-1].apply(lambda m: hyper_bias_init(m))
			
			# print(hn.net[-1])
			# exit()
			self.nets.append(hn)
		
		# exit()

	def forward(self, z_shape,z_color=None):
		'''
		Args:-
			z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

		Returns:
			params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
		'''
		params = OrderedDict()
		for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
			# print(f"name: {name}")
			# print(f"param shape: {param_shape}, {int(torch.prod(torch.tensor(param_shape)))}")
			batch_param_shape = (-1,) + param_shape
			# print(f"batch param shape: {batch_param_shape}")
			if 'color_net' in name:
				params[name] = net(z_color).reshape(batch_param_shape)
			else:
				params[name] = net(z_shape).reshape(batch_param_shape)
			# print(f'name: {name}, param_shape: {param_shape}, params[name].shape: {params[name].shape}')
		return params

############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1e1

	# if hasattr(m, 'bias') and siren:
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1.e1

	# if hasattr(m, 'bias') and siren:
	#     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/fan_in, 1/fan_in)


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	# grab from upstream pytorch branch and paste here for now
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor


def init_weights_trunc_normal(m):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			fan_in = m.weight.size(1)
			fan_out = m.weight.size(0)
			std = math.sqrt(2.0 / float(fan_in + fan_out))
			mean = 0.
			# initialize with the same behavior as tf.truncated_normal
			# "The generated values follow a normal distribution with specified mean and
			# standard deviation, except that values whose magnitude is more than 2
			# standard deviations from the mean are dropped and re-picked."
			_no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
	if type(m) == BatchLinear or type(m) == nn.Linear:
		if hasattr(m, 'weight'):
			nn.init.xavier_normal_(m.weight)


def sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
	y = x.clone()
	y[..., 1::2] = -1 * y[..., 1::2]
	return y


def compl_div(x, y):
	''' x / y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = (a * c + b * d) / (c ** 2 + d ** 2)
	outi = (b * c - a * d) / (c ** 2 + d ** 2)
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out


def compl_mul(x, y):
	'''  x * y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = a * c - b * d
	outi = (a + b) * (c + d) - a * c - b * d
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out

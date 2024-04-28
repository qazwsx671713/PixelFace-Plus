import numpy as np
import torch
import math
import sys
sys.path.append('./pixel_models')
import misc
from torch_utils import misc as torch_misc
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act


def float_dtype():
    return torch.float32

# Flatten all dimensions of a tensor except the fist/last one
def to_2d(x, mode):
    if len(x.shape) == 2:
        return x
    if mode == "last":
        return x.flatten(end_dim = -2) # x.reshape(-1, x.shape[-1])
    else:
        return x.flatten(1) # x.reshape(x.shape[0], element_dim(x))

# Convert tensor to memory format
def format_memory(w, channels_last):
    if channels_last:
        return w.to(memory_format = torch.channels_last)
    return w

# Return a nearest neighbors upsampling kernel
def nearest_neighbors_kernel(device, factor = 2):
    return upfirdn2d.setup_filter([1] * factor, device = device)

# Convert a torch.nn.Parameter to the necessary dtype and apply gain, to be used within 'forward'
def get_param(param, dtype, gain, reorder = False):
    if param is None:
        return None
    if gain != 1 and reorder:
        param = param * gain
    param = param.to(dtype)
    if gain != 1 and not reorder:
        param = param * gain
    return param

# Create a weight variable for a convolution or fully-connected layer. lrmul means learning-rate multiplier 
def get_weight(shape, gain = 1, use_wscale = True, lrmul = 1, channels_last = False):
    fan_in = np.prod(shape[1:])
    he_std = gain / np.sqrt(fan_in)

    # Equalized learning rate and custom learning rate multiplier
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable
    w = torch.randn(shape) * init_std
    w = torch.nn.Parameter(format_memory(w, channels_last))
    return w, runtime_coef

# Create a bias variable for a convolution or fully-connected layer
def get_bias(num_channels, bias_init = 0, lrmul = 1):
    b = torch.nn.Parameter(torch.full([num_channels], np.float32(bias_init)))
    return b, lrmul


# Fully-connected layer act(x@w + b)
@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias = True, act = "linear", gain = 1, lrmul = 1, bias_init = 0):
        super().__init__()
        self.weight, self.w_gain = get_weight([out_channels, in_channels], gain = gain, lrmul = lrmul)
        self.bias,   self.b_gain = get_bias(out_channels, bias_init, lrmul) if bias else None
        self.act = act

    def forward(self, x, _x = None): # _x is added to the signature for backward-compatibility and isn't used
        w = get_param(self.weight, x.dtype, self.w_gain)
        b = get_param(self.bias, x.dtype, self.b_gain)

        if len(x.shape) > 2:
            x = to_2d(x, "first")

        if self.act == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act = self.act)
        return x

# Resent Layer
@persistence.persistent_class
class ResnetLayer(torch.nn.Module):
    def __init__(self, channels, act = "linear", lrmul = 1, sa = False):
        super().__init__()
        self.fc0 = FullyConnectedLayer(channels, channels, act = act, lrmul = lrmul)
        self.fc1 = FullyConnectedLayer(channels, channels, lrmul = lrmul)

    def forward(self, x, _x):
        shape = x.shape
        batch = len(shape) > 2

        if batch:
            x = to_2d(x, "last")
        x = self.fc0(x)
        x = self.fc1(x)
        if batch:
            x = x.reshape(shape)

        x = torch.nn.functional.leaky_relu(x + _x, negative_slope = 0.2) 
        return x

# Multi-layer network with 'layers_num' layers, dimension 'dim', and nonlinearity 'act'.
# Optionally use resnet connections and self-attention.
# If x dimensions are not [batch_size, dim], then turn the input tensor [..., dim] into [-1, dim] 
# so to create one large batch with all the elements from the last axis.
@persistence.persistent_class
class MLP(torch.nn.Module):
    def __init__(self, channels, act, resnet = False, sa = False, batch = True, lrmul = 1, **sa_kwargs):
        super().__init__()

        self.layers_num = int(len(channels) / 2) if resnet else (len(channels) - 1)
        self.out_layer = FullyConnectedLayer(channels[-1], channels[-1], act = act, lrmul = lrmul)
        self.batch = batch
        self.sa = sa

        for idx in range(self.layers_num):
            in_dim = channels[idx]
            out_dim = channels[idx + 1]
            if sa:
                sa_layer = TransformerLayer(dim = in_dim, pos_dim = in_dim, 
                    from_dim = in_dim, to_dim = in_dim, **sa_kwargs)
                setattr(self, f"sa{idx}", sa_layer)

            if resnet:
                layer = ResnetLayer(in_dim, act = act, lrmul = lrmul)
                assert in_dim == out_dim
            else:
                layer = FullyConnectedLayer(in_dim, out_dim, act = act, lrmul = lrmul)
            
            setattr(self, f"l{idx}", layer)

    def forward(self, x, pos = None, mask = None):
        shape = x.shape
        if len(x.shape) > 2:
            x = to_2d(x, "last" if self.batch else "first") 

        for idx in range(self.layers_num):
            _x = x
            if self.sa:
                sa = getattr(self, f"sa{idx}")
                x = sa(from_tensor = x, to_tensor = x, from_pos = pos, to_pos = pos, att_mask = mask.unsqueeze(1))[0]

            layer = getattr(self, f"l{idx}")
            x = layer(x, _x)

        x = self.out_layer(x)

        x = x.reshape(*shape[:-1], -1)
        return x


# Normalization operation used in attention layers. Does not scale back features x (the image)
# with parametrized gain and bias, since these will be controlled by the additive/multiplicative
# integration of as part of the transformer layer (where the latent y will modulate the image features x)
# after x gets normalized, by controlling their scale and bias (similar to the FiLM and StyleGAN approaches).
#
# Arguments:
# - x: [batch_size * num, channels]
# - num: number of elements in the x set (e.g. number of positions WH)
# - integration: type of integration -- additive, multiplicative or both
# - norm: normalization type -- instance or layer-wise
# Returns: normalized x tensor
def att_norm(x, num, integration, norm):
    if norm is None:
        return x

    shape = x.shape
    x = x.reshape([-1, num] + list(shape[1:])).to(float_dtype())

    # instance axis if norm == "instance" and channel axis if norm == "layer"
    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x = x - x.mean(dim = norm_axis, keepdim = True)
    #  X−µ(X)/σ(X)
    #  µ(X)=0
    #  X/σ(X)
    if integration in ["mul", "both"]:
        x = x * torch.rsqrt(torch.square(x).mean(dim = norm_axis, keepdim = True) + 1e-8)

    # return x to its original shape
    x = x.reshape(shape)
    return x

# Dropout and masking
# ----------------------------------------------------------------------------

# Create a random mask of a chosen shape and with probability 'dropout' to be dropped (=0)
def random_dp_binary(shape, dropout, training, device):
    if not training or dropout == 0.0:
        return torch.ones(shape, device = device)
    eps = torch.rand(shape, device = device)
    keep_mask = (eps >= dropout)
    return keep_mask

# Perform dropout
def dropout(x, dp_func, noise_shape = None):
    noise_shape = noise_shape or x.shape
    return dp_func(torch.ones(noise_shape, device = x.device)) * x

# Set a mask for logits (set -Inf where mask is 0)
def logits_mask(x, mask): 
    return x + (1 - mask.to(torch.int32)).to(float_dtype()) * -10000.0

# Positional encoding
# ----------------------------------------------------------------------------

# 2d linear embeddings [size, size, dim] in a [-rng, rng] range where size = grid size and
# dim = the embedding dimension. Each embedding consists of 'num' parts, with each part measuring
# positional similarity along another direction, uniformly spanning the 2d space.
def get_linear_encoding(size, dim, num, rng = 1.0):
    theta = torch.arange(0, math.pi, step = math.pi / num)
    dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim = -1)
    embs = torch.nn.Parameter(torch.rand([num, int(dim / num)]))

    c = torch.linspace(-rng, rng, size)
    x = c.unsqueeze(0).tile([size, 1])
    y = c.unsqueeze(1).tile([1, size])
    xy = torch.stack([x, y], dim = -1)

    lens = (xy.unsqueeze(2) * dirs).sum(dim = -1, keepdim = True)
    emb = (lens * embs).reshape(size, size, dim)
    return emb

# 2d sinusoidal embeddings [size, size, dim] with size = grid size and dim = embedding dimension
# (see "Attention is all you need" paper)
def get_sinusoidal_encoding(size, dim, num = 2):
    # Standard positional encoding in the two spatial w,h directions
    if num == 2:
        c = torch.linspace(-1.0, 1.0, size).unsqueeze(-1)
        i = torch.arange(int(dim / 4)).to(float_dtype())

        peSin = torch.sin(c / (torch.pow(10000.0, 4 * i / dim)))
        peCos = torch.cos(c / (torch.pow(10000.0, 4 * i / dim)))

        peSinX = peSin.unsqueeze(0).tile([size, 1, 1])
        peCosX = peCos.unsqueeze(0).tile([size, 1, 1])
        peSinY = peSin.unsqueeze(1).tile([1, size, 1])
        peCosY = peCos.unsqueeze(1).tile([1, size, 1])

        emb = torch.cat([peSinX, peCosX, peSinY, peCosY], dim = -1)
    # Extension to 'num' spatial directions. Each embedding consists of 'num' parts, with each
    # part measuring positional similarity along another direction, uniformly spanning the 2d space.
    # Each such part has a sinus and cosine components.
    else:
        theta = torch.arange(0, math.pi, math.pi / num)
        dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim = -1)

        c = torch.linspace(-1.0, 1.0, size)
        x = c.unsqueeze(0).tile([size, 1])
        y = c.unsqueeze(1).tile([1, size])
        xy = torch.stack([x, y], dim = -1)

        lens = (xy.unsqueeze(2) * dirs).sum(dim = -1, keepdim = True)

        i = torch.arange(int(dim / (2 * num))).to(float_dtype())
        sins = torch.sin(lens / (torch.pow(10000.0, 2 * num * i / dim)))
        coss = torch.cos(lens / (torch.pow(10000.0, 2 * num * i / dim)))
        emb = torch.cat([sins, coss], dim = -1).reshape(size, size, dim)

    return emb

def get_positional_encoding(
        res, 
        pos_dim,                # Positional encoding dimension
        pos_type = "sinus",     # Positional encoding type: linear, sinus, trainable, trainable2d
        pos_init = "uniform",   # Positional encoding initialization distribution: normal or uniform
        pos_directions_num = 2, # Positional encoding number of spatial directions
        shared = False,         # Share embeddings for x and y axes
        crop_ratio = None,      # Crop the embedding features to the ratio 
        **_kwargs):             # Ignore unrecognized keyword args

    params = []
    initializer = torch.rand if pos_init == "uniform" else torch.randn
    if pos_type == "sinus":
        emb = get_sinusoidal_encoding(res, pos_dim, num = pos_directions_num)
    elif pos_type == "linear":
        emb = get_linear_encoding(res, pos_dim, num = pos_directions_num)
    elif pos_type == "trainable2d":
        emb = torch.nn.Parameter(initializer([res, res, pos_dim]))
        params = [emb]
    else: # pos_type == "trainable"
        xemb = torch.nn.Parameter(initializer([res, int(pos_dim / 2)]))
        yemb = xemb if shared else torch.nn.Parameter(initializer(res, int(pos_dim / 2)))
        params = [xemb, yemb]
        xemb = xemb.unsqueeze(0).tile([res, 1, 1])
        yemb = yemb.unsqueeze(1).tile([1, res, 1])
        emb = torch.cat([xemb, yemb], dim = -1)

    emb = misc.crop_tensor(emb, crop_ratio)
    return emb, params

# Produce trainable embeddings of shape [size, dim], uniformly/normally initialized
def get_embeddings(size, dim, init = "uniform", name = None):
    if size == 0:
        return None
    initializer = torch.rand if init == "uniform" else torch.randn
    emb = torch.nn.Parameter(initializer([size, dim]))
    return emb

# Transpose tensor to scores
def transpose_for_scores(x, num_heads, elem_num, head_size):
    x = x.reshape(-1, elem_num, num_heads, head_size) # [B, N, H, S]
    x = x.permute(0, 2, 1, 3) # [B, H, N, S]
    return x

# Compute attention probabilities: perform softmax on att_scores and dropout
def compute_probs(scores, dp_func):
    # Compute attention probabilities
    probs = torch.nn.functional.softmax(scores, dim = -1) # [B, N, F, T]
    shape = [int(d) for d in probs.shape]
    shape[-2] = 1
    # Dropout over random cells and over random full rows (randomly don't use a 'to' element)
    probs = dropout(probs, dp_func)
    probs = dropout(probs, dp_func, shape)
    return probs

# Gate attention values either row-wise (from) or column-wise so that some of the elements
# in the from/to_tensor will not participate in sending/receiving information, when gate
# value is low for them.
@persistence.persistent_class
class GateAttention(torch.nn.Module):
    def __init__(self, should_gate, dim, pos_dim, num_heads, from_len, to_len, gate_bias = 0):
        super().__init__()
        self.should_gate = should_gate
        self.from_len = from_len
        self.to_len = to_len
        self.num_heads = num_heads
        self.gate_bias = gate_bias

        if should_gate:
            self.gate = FullyConnectedLayer(dim, num_heads)
            self.gate_pos = FullyConnectedLayer(pos_dim, num_heads)

    def forward(self, att_probs, tensor, pos):
        if not self.should_gate:
            return att_probs
        gate = self.gate(tensor)
        if pos is not None:
            gate = gate + self.gate_pos(pos)
        gate = torch.sigmoid(gate + self.gate_bias)
        gate = gate.reshape(-1, self.from_len, self.to_len, self.num_heads).permute(0, 3, 1, 2)
        att_probs = att_probs * gate
        return att_probs

@persistence.persistent_class
class GateCrossModelFusion(torch.nn.Module):
    def __init__(self,
            dim,                                    # The layer dimension
            pos_dim,                                # Positional encoding dimension
            from_len,           to_len,             # The from/to tensors length (must be specified if from/to has 2 dims)
            from_dim,           to_dim,             # The from/to tensors dimensions
            from_gate = False,  to_gate = False,    # Add sigmoid gate on from/to, so that info may not be sent/received
                                                    # when gate is low (i.e. the attention probs may not sum to 1)
            # Additional options
            num_heads           = 1,                # Number of attention heads
            attention_dropout   = 0.12,             # Attention dropout rate
            integration         = "both",           # Feature integration type: additive, multiplicative or both
            norm                = None,             # Feature normalization type (optional): instance, batch or layer

            # k-means options (optional, duplex)
            kmeans              = False,            # Track and update image-to-latents assignment centroids, used in the duplex attention
            kmeans_iters        = 1,                # Number of K-means iterations per transformer layer
            iterative           = False,            # Carry over attention assignments across transformer layers of different resolutions
                                                    # If True, centroids are carried from layer to layer            
            **_kwargs):                             # Ignore unrecognized keyword args

        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.from_len = from_len
        self.to_len = to_len
        self.from_dim = from_dim
        self.to_dim = to_dim
        
        self.num_heads = num_heads
        self.size_head = int(dim / num_heads)

        # We divide by 2 since we apply the dropout layer twice, over elements and over columns
        self.att_dp = torch.nn.Dropout(p = attention_dropout / 2) 

        self.norm = norm
        self.integration = integration        
        
        self.parametric = not iterative
        self.centroid_dim = 2 * self.size_head
        self.kmeans = kmeans
        self.kmeans_iters = kmeans_iters

        # Query, Key and Value mappings
        self.to_queries = FullyConnectedLayer(from_dim, dim)
        self.to_keys    = FullyConnectedLayer(to_dim, dim)
        self.to_values  = FullyConnectedLayer(to_dim, dim)

        # Positional encodings
        self.from_pos_map = FullyConnectedLayer(pos_dim, dim)
        self.to_pos_map   = FullyConnectedLayer(pos_dim, dim)

        # Attention gates
        self.to_gate_attention   = GateAttention(to_gate, dim, pos_dim, num_heads, from_len = 1, to_len = to_len)
        self.from_gate_attention = GateAttention(from_gate, dim, pos_dim, num_heads, from_len = from_len, to_len = 1, gate_bias = 1)

        # Features Integration
        control_dim = (2 * self.dim) if self.integration == "both" else self.dim 
        self.modulation = FullyConnectedLayer(self.dim, control_dim)

        self.c_mlp = MLP([dim]+[dim, dim], act = 'lrelu')
        self.ffn = MLP([dim]+[dim, dim], act = 'lrelu')

    # Validate transformer input shape for from/to_tensor and reshape to 2d
    def process_input(self, t, t_pos, name):
        shape = t.shape
        t_len = getattr(self, f"{name}_len")
        t_dim = getattr(self, f"{name}_dim")

        # from/to_tensor should be either 2 or 3 dimensions. If it's 3, then t_len should be specified.
        if len(shape) > 3:
            misc.error(f"Transformer {name}_tensor has {shape} shape. should be up to 3 dims.")
        elif len(shape) == 3:
            torch_misc.assert_shape(t, [None, t_len, t_dim])
            batch_size = shape[0]
        else:
            # Infer batch size for the 2-dims case
            torch_misc.assert_shape(t, [None, t_dim])
            batch_size = int(shape[0] / t_len)

        # Reshape tensors to 2d
        t = to_2d(t, "last")
        if t_pos is not None:
            t_pos = to_2d(t_pos, "last")
            torch_misc.assert_shape(t_pos, [t_len, self.pos_dim])
            t_pos = t_pos.tile([batch_size, 1])

        return t, t_pos, shape

    # Normalizes the 'tensor' elements, and then integrate the new information from
    # 'control' with 'tensor', where 'control' controls the bias/gain of 'tensor'.
    # norm types: batch, instance, layers
    # integration types: add, mul, both
    def integrate(self, tensor, tensor_len, control): # integration, norm
        # Normalize tensor
        tensor = att_norm(tensor, tensor_len, self.integration, self.norm)

        # Compute gain/bias
        bias = gain = control = self.modulation(control)

        if self.integration == "both":
            shape=control.shape
            gain,bias = torch.split(control, shape[1]//2, dim = -1)

        # Modulate the bias/gain of 'tensor'
        if self.integration != "add":
            tensor = tensor * (gain + 1)
        if self.integration != "mul":
            tensor = tensor + bias

        return tensor

    def forward(self, from_tensor, to_tensor, from_pos, to_pos, 
            att_vars = None, att_mask = None, hw_shape = None):
        # Validate input shapes and map them to 2d
        from_tensor, from_pos, from_shape = self.process_input(from_tensor, from_pos, "from")
        to_tensor,   to_pos,   to_shape   = self.process_input(to_tensor, to_pos, "to")

        att_vars = att_vars or {}
        to_from = att_vars.get("centroid_assignments")

        # Compute queries, keys and values
        queries = self.to_queries(from_tensor)
        keys    = self.to_keys(to_tensor)
        values  = self.to_values(to_tensor)
        _queries = queries

        # Add positional encodings to queries and keys
        if from_pos is not None:
            queries = queries + self.from_pos_map(from_pos)
        if to_pos is not None:
            keys = keys + self.to_pos_map(to_pos)

        # Reshape queries, keys and values, and then compute att_scores
        queries = transpose_for_scores(queries, self.num_heads, self.from_len, self.size_head)  # [B, N, F, H]
        keys = transpose_for_scores(keys,    self.num_heads, self.to_len,   self.size_head)  # [B, N, T, H]
        values = transpose_for_scores(values,  self.num_heads, self.to_len,   self.size_head)  # [B, N, T, H]
        ###########
        # Q*K^T
        ##########
        att_scores = queries.matmul(keys.permute(0, 1, 3, 2)) # [B, N, F, T]
        att_probs = None

        for i in range(self.kmeans_iters):
            # Scale attention scores given head size (see BERT)
            att_scores = att_scores / math.sqrt(float(self.size_head))
            # (optional, not used by default)
            # Mask attention logits using att_mask (to mask some components)
            if att_mask is not None:
                att_scores = logits_mask(att_scores, att_mask.unsqueeze(1))
            # Turn attention logits to probabilities (softmax + dropout)
            ############
            # softmax
            ############
            # out_att_probs = att_probs.clone()
            att_probs = compute_probs(att_scores, self.att_dp)
        # Gate attention values for the from/to elements
        att_probs = self.to_gate_attention(att_probs, to_tensor, to_pos)
        att_probs = self.from_gate_attention(att_probs, from_tensor, from_pos)


        # Compute weighted-sum of the values using the attention distribution
        #############
        # softmax*V
        #############
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        control = control.reshape(-1, self.dim) # [B*F, N*H]
        
        control = self.c_mlp(control)  # [B*F, N*H]

        # # This newly computed information will control the bias/gain of the new from_tensor
        # from_tensor = self.integrate(from_tensor, self.from_len, control)
        
        from_tensor = control + self.ffn(control)  # [B*F, N*H]
        # from ipdb import set_trace; set_trace()

        # Reshape from_tensor to its original shape (if 3 dimensions)
        if len(from_shape) > 2:
            from_tensor = from_tensor.reshape(from_shape)

        if hw_shape is not None:
            if len(hw_shape)==2:
                att_probs = att_probs.reshape(-1, *hw_shape, self.to_len).permute(0, 3, 1, 2) # [NCHW]
            else:
                att_probs = att_probs.reshape(-1, *hw_shape, self.to_len) # [NCHW]

        return from_tensor, att_scores, {"centroid_assignments": to_from}

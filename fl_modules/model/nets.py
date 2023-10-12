import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomposedDense(nn.Module):
    """ Custom dense layer that decomposes parameters into sigma and psi.
  
    Base code is referenced from official TensorFlow code (https://github.com/tensorflow/tensorflow/)

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self,
               in_features,
               out_features,
               activation=None,
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               sigma=None,
               psi=None,
               l1_thres=None):
        super(DecomposedDense, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = use_bias
        self.sigma = sigma
        self.psi = psi
        self.l1_thres = l1_thres

        # Initialize weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        if self.training:
            sigma = self.sigma
            psi = self.psi 
        else: 
            sigma = self.sigma
            hard_threshold = torch.gt(torch.abs(self.psi), self.l1_thres).float()
            psi = self.psi * hard_threshold
        
        # Decomposed Kernel
        theta = sigma + psi

        outputs = F.linear(inputs, theta, bias=self.bias)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class DecomposedConv(nn.Conv2d):
    """ Custom conv layer that decomposes parameters into sigma and psi.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device = None,
        dtype = None,
        sigma: nn.Parameter = None,
        psi: nn.Parameter = None,
        l1_thres = None,
    ):
        """
        Args:
            sigma: nn.Parameter, shape (out_channels, in_channels, kernel_size, kernel_size)
            psi: nn.Parameter, shape (out_channels, in_channels, kernel_size, kernel_size)
            l1_thres: float, threshold for hard thresholding
        """
        super(DecomposedConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        
        self.psi = psi
        self.sigma = sigma
        self.l1_thres = l1_thres

    def forward(self, inputs):
        if self.training:
            sigma = self.sigma
            psi = self.psi 
        else: 
            sigma = self.sigma
            hard_threshold = torch.gt(torch.abs(self.psi), self.l1_thres).float()
            psi = self.psi * hard_threshold
        
        # Decomposed Kernel
        theta = sigma + psi

        outputs = F.conv2d(inputs, theta, stride=self.stride, padding=self.padding, dilation=self.dilation)

        if self.bias is not None:
            outputs += self.bias.view(1, -1, 1, 1)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

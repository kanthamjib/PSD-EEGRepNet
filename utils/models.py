"""
PSD-EEGRepNet - Model Definitions
=================================
This module provides *only* the architectural components used in our paper
"Power-Spectral Density-Driven CNN with RepBlock for Motor-Imagery EEG".

Notes for Reproducibility
------------------------
* **Inference-time re-parameterisation** - Although each RepBlock can be
  structurally compressed into a single convolution (cf. RepVGG v1), the
  current release keeps the multi-branch structure during evaluation because the
  published results were obtained under that setting.  The helper method
  `RepBlock.switch_to_deploy` is intentionally left unimplemented, pending
  validation and incorporation of re-parameterisation techniques in subsequent studies.
* **Computational layout** - All convolutions operate on inputs shaped as
  `(𝑁, 1, 𝐻, 𝑊)` after PSD extraction. (Please see more detail in this Model architecture's Paper)

References
----------
.. [1] X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, and J. Sun,
       “RepVGG: Making VGG-style ConvNets Great Again,” in
       *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
       Los Alamitos, CA, USA: IEEE Computer Society, pp. 13728-13737, June 2021.
"""

import torch.nn as nn

# --------------------------------------------------------------------------------
# Section 1: RepBlock
# --------------------------------------------------------------------------------
class RepBlock(nn.Module):
    """
    Lightweight multi-branch building block inspired by RepVGG [1]_.

    During training, the block consists of three parallel branches:

        main_conv (kx1)   ┐
        conv_1x1  (1x1)   ├──►  Σ  ──► ReLU
        identity          ┘

    The identity branch is enabled only if input and output channel counts match and
    the stride is unity.

    At inference time, the branches can theoretically be merged into a single convolution
    for computational efficiency, although this re-parameterisation step is intentionally
    not implemented in the current release due to a lack of empirical validation in our
    experiments.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple[int, int], default (3, 1)
        Kernel dimensions tailored for EEG frequency-domain processing.
    stride : tuple[int, int], default (1, 1)
        Stride size shared across branches.
    deploy : bool, default False
        When True, instantiate the block in its re-parameterised single-branch form.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 1), stride=(1, 1), deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_c = in_channels
        self.out_c = out_channels
        self.k = kernel_size
        self.s = stride

        if deploy:
            # Single‑branch representation
            self.reparam_conv = nn.Conv2d(in_channels, out_channels,
                                          kernel_size=self.k,
                                          stride=self.s,
                                          padding=0,
                                          bias=True)
        else:
            # Branch 1 - main convolution
            self.conv_main = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=self.k,
                                        stride=self.s,
                                        padding=0,
                                        bias=True)
            # Branch 2 - 1×1 convolution
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=(1, 1),
                                        stride=self.s,
                                        padding=0,
                                        bias=True)
            # Branch 3 - identity
            self.use_identity = (in_channels == out_channels) and (stride == (1, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam_conv(x))

        main_out = self.conv_main(x)
        branch1 = self.conv_1x1(x)

        desired_h = main_out.shape[2]
        branch1 = branch1[:, :, :desired_h, :]

        out = main_out + branch1
        if self.use_identity:
            id_branch = x
            if x.shape[2] != desired_h:
                id_branch = x[:, :, :desired_h, :]
            out += id_branch
        return self.relu(out)

    def switch_to_deploy(self):
        """This method is intentionally unimplemented pending validation in future work."""
        pass

# --------------------------------------------------------------------------------
# Section 2: PSD‑EEGRepNet
# --------------------------------------------------------------------------------
class PSDEEGRepNet(nn.Module):
    """Convolutional neural network architecture for motor-imagery EEG classification
    utilizing power spectral density (PSD) representations.

    The network architecture comprises sequential RepBlock-BatchNorm-ReLU layers
    interleaved with max-pooling operations to reduce dimensionality along the frequency
    axis. Fully connected layers form the classifier stage.

    Parameters
    ----------
    num_classes : int, default = 3
        Number of target classes.
    deploy : bool, default = False
        Passed to individual RepBlocks, set to True only after explicitly re-parameterising
        each block.
    """

    def __init__(self, num_classes=3, deploy=False):
        super(PSDEEGRepNet, self).__init__()

        self.block1 = RepBlock(1,   25, kernel_size=(4, 1), stride=(1, 1), deploy=deploy)
        self.bn1 = nn.BatchNorm2d(25)

        self.block2 = RepBlock(25, 100, kernel_size=(4, 1), stride=(1, 1), deploy=deploy)
        self.bn2 = nn.BatchNorm2d(100)

        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.block3 = RepBlock(100, 200, kernel_size=(3, 1), stride=(1, 1), deploy=deploy)
        self.bn3 = nn.BatchNorm2d(200)

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # ↓ freq 2×

        self.block4 = RepBlock(200, 200, kernel_size=(1, 1), stride=(1, 1), deploy=deploy)
        self.bn4 = nn.BatchNorm2d(200)

        self.fc1 = nn.Linear(200 * 1 * 30, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.block1(x)))
        x = self.relu(self.bn2(self.block2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.block3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.block4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


    def switch_to_deploy(self):
        """Invoke the re-parameterisation method on each RepBlock."""
        for name in ["block1", "block2", "block3", "block4"]:
            getattr(self, name).switch_to_deploy()

# --------------------------------------------------------------------------------
# Public Factory Method
# --------------------------------------------------------------------------------
def get_model(n_classes: int, deploy: bool = False):
    """Instantiate and return a PSD-EEGRepNet model instance."""
    return PSDEEGRepNet(num_classes=n_classes, deploy=deploy)

# PSD-EEGRepNet: A CNN Architecture with Multi-Branch RepBlocks for Power Spectral Density-Based Motor Imagery EEG Classification in BCI

**PSD-EEGRepNet** is a lightweight CNN-based neural network architecture utilizing RepBlocks designed specifically for Motor Imagery (MI) EEG signal classification. The model leverages Power Spectral Density (PSD) features, efficiently extracted using Welch's method, to classify EEG signals into distinct motor imagery tasks.

The primary goal of PSD-EEGRepNet is to `achieve robust classification accuracy while maintaining low computational complexity during model training and inference`. It is particularly optimized for use in Brain-Computer Interface (BCI) applications, providing a good balance between performance and computational efficiency.

## Key Features

- **Efficient Design**: Lightweight multi-branch convolutional architecture inspired by RepVGG.

- **PSD-Based Features**: Uses Welch’s method for extracting Power Spectral Density features.

- **High Performance**: Evaluated with cross-validation showing promising accuracy on PhysioNet datasets.

- **Rapid Training**: Quick training times make iterative model development straightforward.

## Experimental Results

PSD-EEGRepNet was evaluated on the `PhysioNet Motor Imagery dataset` comprising EEG data from 10 subjects (For more information, please feel free to study: https://physionet.org/content/eegmmidb/1.0.0/). Using a 5-fold stratified cross-validation scheme and two different data overlap conditions (80% and 90%), the model demonstrated:

**Accuracy**:

- 80.38% (80% data overlap)

- 85.69% (90% data overlap)

**Model Complexity**:

- Parameters: ~6.32 million

- Computational Cost: ~20 MFLOPs

**Training Efficiency**:

- Average training time per fold: 8.8 seconds (80% overlap), 20.4 seconds (90% overlap)

These results underline the potential of PSD-EEGRepNet as a practical and efficient solution for EEG-based motor imagery classification tasks.

## Requirements
This project requires the following packages and dependencies to run successfully. It supports both PyTorch and TensorFlow environments (CUDA 11.8 recommended for GPU acceleration).

(Full details provided in `requirements.txt`)

## Installation

To install PSD-EEGRepNet, clone this repository and install the required dependencies:

```bash
git clone https://github.com/kanthamjib/PSD-EEGRepNet.git
cd PSD-EEGRepNet
pip install -r requirements.txt
```

## Usage

### Running the main pipeline
```bash
python main.py
```

### Configuration
Adjust configurations (sampling rate, frequency bands, training epochs, etc.) by editing the file:
```bash
config/default.yaml
```

## Citation

If you find this repository useful in your research, please consider citing: https://ieeexplore.ieee.org/document/11113794

K. Thangthong, F. Asadi and S. Tungjitkusolmun, "PSD-EEGRepNet: A CNN Architecture with Multibranch RepBlocks for Power Spectral Density-Based Motor Imagery EEG Classification in BCI," 2025 17th Biomedical Engineering International Conference (BMEiCON), Chiang Mai, Thailand, 2025, pp. 1-5, doi: 10.1109/BMEiCON66226.2025.11113794.

```bibtex
@inproceedings{PSD_EEGRepNet_2025,
  author    = {Kantham Thangthong and Fawad Asadi and Supan Tungjitkusolmun},
  title     = {{PSD-EEGRepNet: A CNN Architecture with Multibranch RepBlocks for Power Spectral Density-Based Motor Imagery EEG Classification in BCI}},
  booktitle = {Proceedings of the 2025 17th Biomedical Engineering International Conference (BMEiCON)},
  year      = {2025},
  month     = {July},
  pages     = {1--5},
  address   = {Chiang Mai, Thailand},
  publisher = {IEEE},
  doi       = {10.1109/BMEiCON66226.2025.11113794},
  url       = {https://ieeexplore.ieee.org/document/11113794}
}



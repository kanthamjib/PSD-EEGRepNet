import numpy as np

# -----------------------------------------------------------------
# 1)  Define dataset dimensions
# -----------------------------------------------------------------
n_epochs   = 60          # small, so the test finishes in < 1 min on CPU
n_channels = 19          # the 19-ch montage you used when computing PSD
n_times    = 640         # 4 s * 160 Hz  → 640 samples
n_classes  = 3           # MI tasks: e.g. left, right, rest

# -----------------------------------------------------------------
# 2)  Generate fake raw windows  (white noise ─ realistic enough for a dry run)
# -----------------------------------------------------------------
rng = np.random.default_rng(seed=42)
windows_data  = rng.standard_normal(size=(n_epochs, n_channels, n_times))

# -----------------------------------------------------------------
# 3)  Dummy labels  (balanced, string form to match LabelEncoder)
# -----------------------------------------------------------------
labels_int    = np.repeat(np.arange(n_classes), n_epochs // n_classes)
rng.shuffle(labels_int)
windows_label = np.array(["T"+str(i) for i in labels_int])

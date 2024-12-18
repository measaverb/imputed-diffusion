# Preprocessing Daphnet

Previous works for detecting Freezing of Gait (FoG) were focused on supervised or denoising methods. Most of them evaluated their methods by randomly sampling for the training and test sets. However, the method in this repository detects anomalous data by thresholding the reconstructed data, thereby, the usage of data should change.

For methods that detect anomalies by thresholding the reconstructed data, samples in the training and test sets should change. The training set should only contain normal samples, while the test set can contain both normal and abnormal samples. There should be manual manipulation. As the pre-defined window length is 128 (2 seconds * 64 Hz), I’ve split the entire data by 128. If more than a single point is anomalous in the specific segment, it is assumed to be an anomalous segment. After this classification process, the training and test sets have 70% and 30% of segments, respectively, but the training set only contains normal segments. As the data usage varies from previous works, the direct comparison is not plausible.

The preprocessing contains 3-Sigma outlier removal and min-max normalization.


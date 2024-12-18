# trunk-ignore(bandit/B403)
import pickle
import random
from glob import glob

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def three_sigma_outlier_removal(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    data_cleaned = np.where(
        (data < lower_bound) | (data > upper_bound), np.median(data, axis=0), data
    )
    return data_cleaned


def normalise_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)


if __name__ == "__main__":
    # data format: (timestamp, x0, y0, z0, x1, y1, z1, x2, y2, z2, label)
    all_train_features = None
    all_train_labels = None
    all_test_features = None
    all_test_labels = None

    items = glob("raw/*.txt")
    items.sort()
    for item in items:
        print(f"Processing {item}...")

        data = np.genfromtxt(item, delimiter=" ")
        features = data[:, 1:-1]
        features = three_sigma_outlier_removal(features)
        features = normalise_data(features)
        labels = data[:, -1]
        print("Before discarding 0:", features.shape, labels.shape)

        mask = labels != 0
        features = features[mask]
        labels = labels[mask]
        features = features[: features.shape[0] // 128 * 128]
        labels = labels[: labels.shape[0] // 128 * 128]
        print("After discarding 0:", features.shape, labels.shape)

        normal_slice = []
        abnormal_slice = []
        for i in range(0, len(labels), 128):
            slice_labels = labels[i : i + 128]
            if np.all(slice_labels == 1):
                normal_slice.append(i)
            else:
                abnormal_slice.append(i)
        print(
            f"Normal slices: {len(normal_slice)}, abnormal slices: {len(abnormal_slice)}"
        )

        try:
            test_normal_slice = random.sample(
                normal_slice,
                int((len(normal_slice) + len(abnormal_slice)) * 0.3)
                - len(abnormal_slice),
            )
        except ValueError:
            print("Not enough normal slices to form a test set")
            test_normal_slice = []

        train_slice = list(set(normal_slice) - set(test_normal_slice))
        test_slice = test_normal_slice + abnormal_slice
        print(f"Train slices: {len(train_slice)} Test slices: {len(test_slice)}")

        train_features = np.vstack([features[i : i + 128] for i in train_slice])
        train_labels = np.hstack([labels[i : i + 128] - 1 for i in train_slice])
        test_features = np.vstack([features[i : i + 128] for i in test_slice])
        test_labels = np.hstack([labels[i : i + 128] - 1 for i in test_slice])

        if all_train_features is None:
            all_train_features = train_features
            all_train_labels = train_labels
            all_test_features = test_features
            all_test_labels = test_labels
        else:
            all_train_features = np.vstack((all_train_features, train_features))
            all_train_labels = np.hstack((all_train_labels, train_labels))
            all_test_features = np.vstack((all_test_features, test_features))
            all_test_labels = np.hstack((all_test_labels, test_labels))

    print("Final shapes:")
    print(all_train_features.shape, all_train_labels.shape)
    print(all_test_features.shape, all_test_labels.shape)

    # Save the train and test sets
    with open("processed/daphnet_train.pkl", "wb") as f:
        pickle.dump((all_train_features), f)
    with open("processed/daphnet_test.pkl", "wb") as f:
        pickle.dump((all_test_features), f)
    with open("processed/daphnet_test_label.pkl", "wb") as f:
        pickle.dump((all_test_labels), f)

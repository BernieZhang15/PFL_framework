import random
import numpy as np
np.float_ = np.float64
from imagecorruptions import corrupt


SEVERITIES = [1, 2, 3]
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]


def corrupt_images(train_data, test_data):

    corrupted_train_data = []
    corrupted_test_data = []

    for train_client, test_client in zip(train_data, test_data):

        severity, corruption = random.sample(SEVERITIES, 1)[0], random.sample(CORRUPTIONS, 1)[0]

        train_client_data = train_client['x']
        train_client_label = train_client['y']
        test_client_data = test_client['x']
        test_client_label = test_client['y']

        corrupted_train_client_data = []
        corrupted_test_client_data = []

        for i in range(train_client_data.shape[0]):
            image = (train_client_data[i] * 255).transpose(1, 2, 0).astype(np.uint8)
            corrupted_image = np.expand_dims(corrupt(image, severity=severity, corruption_name=corruption).transpose((2, 0, 1)), axis=0)
            normalized_corrupted_image = (corrupted_image.astype(np.float32)) / 255.0
            corrupted_train_client_data.append(normalized_corrupted_image)

        for i in range(test_client_data.shape[0]):
            image = (test_client_data[i] * 255).transpose(1, 2, 0).astype(np.uint8)
            corrupted_image = np.expand_dims(corrupt(image, severity=severity, corruption_name=corruption).transpose((2, 0, 1)), axis=0)
            normalized_corrupted_image = (corrupted_image.astype(np.float32)) / 255.0
            corrupted_test_client_data.append(normalized_corrupted_image)

        corrupted_train_client_data = np.concatenate(corrupted_train_client_data, axis=0)
        corrupted_test_client_data = np.concatenate(corrupted_test_client_data, axis=0)

        corrupted_train_data.append({"x": corrupted_train_client_data, "y": train_client_label})
        corrupted_test_data.append({"x": corrupted_test_client_data, "y": test_client_label})

    return corrupted_train_data, corrupted_test_data



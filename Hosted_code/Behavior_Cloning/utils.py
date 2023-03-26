def prepare_data(file_path):
    import os
    import numpy as np
    from PIL import Image

    train_images_path = os.path.join(file_path, 'train_images')
    test_images_path = os.path.join(file_path, 'test_images')

    train_paths = os.listdir(train_images_path)
    test_paths = os.listdir(test_images_path)

    train_labels = np.load(file_path + 'train_labels.npy?raw=true', allow_pickle=True)
    test_labels = np.load(file_path + 'test_labels.npy?raw=true', allow_pickle=True)

    def find_steering_angle(image_name, labels_array):
        for row in labels_array:
            if row[0] == image_name:
                return float(row[1])
        return None

    train_images = []
    train_angles = []
    for image_name in train_paths:
        img = Image.open(os.path.join(train_images_path, image_name))
        img_array = np.array(img)
        train_images.append(img_array)
        angle = find_steering_angle(image_name, train_labels)
        train_angles.append(angle)

    test_images = []
    test_angles = []
    for image_name in test_paths:
        img = Image.open(os.path.join(test_images_path, image_name))
        img_array = np.array(img)
        test_images.append(img_array)
        angle = find_steering_angle(image_name, test_labels)
        test_angles.append(angle)

    train_images = np.array(train_images, dtype=object)
    train_angles = np.array(train_angles)
    test_images = np.array(test_images, dtype=object)
    test_angles = np.array(test_angles)

    print(f"Training images: {train_images.shape}")
    print(f"Test images: {test_images.shape}")
    print(f"Train angles: {train_angles.shape}")
    print(f"Test angles: {test_angles.shape}")

    return train_images, train_angles, test_images, test_angles

from torch.utils.data import Dataset
class DriveDataset(Dataset):
    def __init__(self, images, targets):
        self.images_list = images
        self.target_list = targets
        assert (len(self.images_list) == len(self.target_list))
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, key):
        image_idx = self.images_list[key]
        target_idx = self.target_list[key]
        # Correct datatype here
        return [image_idx.astype(np.float32), target_idx.astype(np.float32)]


def plot_sample_images(images, labels):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 3, figsize=(16, 5), dpi=100)

    random_9 = np.random.randint(0, len(images), 9)

    for i in range(3):
        for j in range(3):
            index = random_9[i * 3 + j]
            img = images[index]
            angle = labels[index]

            # Ensure the image is a NumPy array with an appropriate data type
            img_array = np.array(img).astype(np.uint8)

            ax[i][j].imshow(img_array)
            ax[i][j].set_title(f"Steering angle: {angle:.2f}")
            ax[i][j].axis('off')  # Remove x and y axis labels
            
    plt.show()
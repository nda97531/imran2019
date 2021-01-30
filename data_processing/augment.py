import numpy as np


def augment_inertial(ine_data, jiter_factor=500):
    """
    :param ine_data: (timestep, 3-axis gyro)
    :param jiter_factor:
    :return: (timestep, 3-axis gyro) with Gaussian noise
    """
    augmented_data = np.empty_like(ine_data)
    for i in range(ine_data.shape[-1]):
        data = ine_data[:, i]
        dataUnique = np.unique(data)
        dataDiff = np.abs(np.diff(dataUnique))
        smallestDiff = np.min(dataDiff)
        scaleFactor = 0.2 * jiter_factor * smallestDiff
        augmented_data[:, i] = data + scaleFactor * np.random.randn(data.shape[0])

    return augmented_data


def augment_skeleton(ske_data, degree_range=5):
    """
    :param ske_data: (timestep, joint, coordinate)
    :param degree_range:
    :return: (timestep, joint, coordinate) rotated
    """
    org_dim = len(ske_data.shape)

    if org_dim == 2:
        ske_data = ske_data.reshape([ske_data.shape[0], -1, 3])

    theta = np.random.uniform(-degree_range, degree_range, size=1) / 180 * np.pi
    gamma = np.random.uniform(-degree_range, degree_range, size=1) / 180 * np.pi

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    Ry = np.array([[np.cos(gamma), 0, np.sin(gamma)],
                   [0, 1, 0],
                   [-np.sin(gamma), 0, np.cos(gamma)]])
    R = np.dot(Rx, Ry)
    augmented_data = np.empty(ske_data.shape)
    for i in range(ske_data.shape[0]):
        augmented_data[i] = np.dot(R, ske_data[i].T).T

    if org_dim == 2:
        augmented_data = augmented_data.reshape([augmented_data.shape[0], -1])
    return augmented_data

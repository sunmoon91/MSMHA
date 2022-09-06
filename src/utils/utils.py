import numpy as np
import SimpleITK as sitk





def load_data_train_3D(data_dir, cv):
    w1 = 256
    w2 = 256
    w3 = 20
    print('cross-validation: ', cv)
    idx = np.concatenate([range(0, (cv - 1) * 24), range((cv) * 24, 120)], 0)

    train_data = np.zeros([96, w1, w2, w3, 1])
    train_label = np.zeros([96, w1, w2, w3, 1])

    for i in range(0, 96):
        ss = int(idx[i])
        img_dir = data_dir + 'stack-' + str(ss + 1) + '.nii.gz'
        print(img_dir)
        I = sitk.ReadImage(img_dir)
        img = np.array(sitk.GetArrayFromImage(I))
        img = img.transpose((2, 1, 0))
        img = np.reshape(img, [-1, w1, w2, w3, 1])
        train_data[i, :, :, :] = img
        lab_dir = data_dir + 'label-' + str(ss + 1) + '.nii.gz'
        print(lab_dir)
        I = sitk.ReadImage(lab_dir)
        img = np.array(sitk.GetArrayFromImage(I))
        img = img.transpose((2, 1, 0))
        img = np.reshape(img, [-1, w1, w2, w3, 1])
        train_label[i, :, :, :, :] = img

    return train_data, train_label


def load_data_eval_3D(data_dir, cv):
    w1 = 256
    w2 = 256
    w3 = 20
    print('cross-validation: ', cv)
    idx = range((cv - 1) * 24, cv * 24)

    test_data = np.zeros([24, w1, w2, w3, 1])
    test_label = np.zeros([24, w1, w2, w3, 1])

    for i in range(0, 24):
        ss = int(idx[i])
        img_dir = data_dir + 'stack-' + str(ss + 1) + '.nii.gz'
        print(img_dir)
        I = sitk.ReadImage(img_dir)
        img = np.array(sitk.GetArrayFromImage(I))
        img = img.transpose((2, 1, 0))
        img = np.reshape(img, [-1, w1, w2, w3, 1])
        test_data[i, :, :, :] = img
        lab_dir = data_dir + 'label-' + str(ss + 1) + '.nii.gz'
        print(lab_dir)
        I = sitk.ReadImage(lab_dir)
        img = np.array(sitk.GetArrayFromImage(I))
        img = img.transpose((2, 1, 0))
        img = np.reshape(img, [-1, w1, w2, w3, 1])
        test_label[i, :, :, :, :] = img

    return test_data, test_label



def get_batch(ite, batchNumber, data, label_map):
    idx1 = ite * batchNumber
    idx2 = (ite + 1) * batchNumber
    data_batch = data[idx1:idx2, :, :, :, :]
    label_batch = label_map[idx1:idx2, :, :, :, :]
    return data_batch, label_batch


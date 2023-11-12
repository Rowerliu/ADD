import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

def image_fourier_trans(image, filter='HP'):

    def create_filter_mask(img, type, threshold, scale):
        # 创建滤波器掩码
        rows, cols, _ = img.shape
        crow, ccol = rows // 2, cols // 2
        if type == 'HP':
            mask = np.ones((rows, cols), np.uint8)
            mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
        elif type == 'LP':
            mask = np.ones((rows, cols), np.uint8) * scale
            mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1.0
        else:
            mask = np.ones((rows, cols), np.uint8)
        return mask


    def image_gray_fft(img, filter=None, threshold=10, scale=0):

        # 2. 转换为灰度图像
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = img_gray / 255
        rows, cols = img_gray.shape

        # 3. 执行二维傅里叶变换
        dft = np.fft.fft2(img_gray)
        dft_shift = np.fft.fftshift(dft)

        # 创建高通和低通滤波器掩码
        mask_h = create_filter_mask(img, 'HP', threshold, scale)  # 高通滤波

        # 3.3 滤波
        if filter == 'HP':
            mask = mask_h
        else:
            mask = np.ones((rows, cols), np.uint8)

        dft_shift = dft_shift * mask
        # magnitude = np.log(np.abs(dft_shift))
        magnitude = np.abs(dft_shift)

        return magnitude


    def cal_high_freq(magnitude, threshold):

        center = int(magnitude.shape[0] / 2)
        magnitude_left = magnitude[center, int(center - 10*threshold):int(center - threshold)]
        magnitude_right = magnitude[center, int(center + 1*threshold):int(center + 10*threshold)]
        magnitude_all = np.concatenate((magnitude_left, magnitude_right))
        magnitude_average = np.average(magnitude_all)

        return magnitude_average

    # 读取图像
    if isinstance(image, torch.Tensor):
        # image = ((image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        image = image.cpu().numpy()
        image = image[0].transpose(1, 2, 0)

    magnitude = image_gray_fft(image, filter=filter, threshold=1, scale=0.5)
    high_freq_magnitude = cal_high_freq(magnitude, threshold=1)
    return high_freq_magnitude

def get_adaptive_list(total_list, amount):

    list_min = min(total_list)
    list_max = max(total_list)
    value_range = list_max - list_min

    value_list = []
    for i in range(1, amount+1):
        value = list_min + value_range * i / amount
        value_list.append(value)

    positions = []
    position_prev = 0
    position_find = True

    for value in value_list:
        search_time = 0
        value_position_appropriate = False
        while (value_position_appropriate == False) and (position_find == True):
            closest = min(total_list, key=lambda x: abs(x - value))
            position = total_list.index(closest)
            if position > position_prev:
                value_position_appropriate = True
                position_prev = position
            else:
                total_list[position] = 0
            if search_time > 100:
                position_find = False
            search_time = search_time + 1

        positions.append(position)

    if position_find == False:
        length = len(total_list)
        gap = length // amount
        positions = [i * gap for i in range(1, amount + 1)]
        positions[-1] = positions[-1] - 1

    return positions
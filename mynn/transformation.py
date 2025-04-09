import numpy as np
import cv2

class ImageRandomFlip:
    def __init__(self, prob=0.5, shape_location=(1, 2), direction="horizontal"):
        '''
        随机对图片进行翻转
        prob: 进行翻转的概率
        shape_location: 输入图像中哪个维度表示H，W
        direction: 翻转的方向, optional: 
        '''
        self.prob = prob
        self.shape_location = shape_location
        self.direction = direction
    
    def __call__(self, image):
        p = np.random.rand()
        if p > self.prob:
            return image
        else:
            if self.direction == "horizontal":
                target_location = self.shape_location[1]
            else:
                target_location = self.shape_location[0]
            image = np.flip(image, axis=target_location)
            return image

class ImageRandomCrop:
    def __init__(self, prob=0.5, shape_location=(1, 2), area_cut=0.2):
        '''
        随机对图片进行裁切
        prob: 进行裁切的概率
        range: 最多损失的面积百分比
        '''
        self.prob = prob
        self.shape_location = (1, 2)
        self.edge_cut = np.sqrt(area_cut)
    
    def __call__(self, image):
        indices = [slice(None)] * image.ndim
        p = np.random.rand()
        if p > self.prob:
            return image
        else:
            H = image.shape[self.shape_location[0]]
            W = image.shape[self.shape_location[1]]
            new_H = H * ( 1 - self.edge_cut + self.edge_cut * np.random.rand())
            new_W = W * ( 1 - self.edge_cut + self.edge_cut * np.random.rand())
            start_H = (H - new_H) * np.random.rand()
            start_W = (W - new_W) * np.random.rand()
            end_H = int(start_H + new_H)
            end_W = int(start_W + new_W)
            start_H = int(start_H)
            start_W = int(start_W)
            indices[self.shape_location[0]] = slice(start_H, end_H+1)
            indices[self.shape_location[1]] = slice(start_W, end_W+1)
            new_image = image[tuple(indices)]
            new_image = np.transpose(cv2.resize(np.transpose(new_image, (1, 2, 0)), (H, W)), (2, 0, 1))
            return new_image


def adjust_brightness(image, brightness_factor):
    table = np.array([i * brightness_factor for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, table)

# def adjust_contrast(image, contrast_factor):
#     table = np.array([i * contrast_factor for i in range(256)]).clip(0, 255).astype(np.uint8)
#     return cv2.LUT(image, table)

def adjust_saturation(image, saturation_factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

class ImageRandomJitter:
    def __init__(self, brightness=0.2, saturation=0.2):
        self.brightness = brightness
        self.saturation = saturation

    def __call__(self, image):
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            image = np.transpose(adjust_brightness(np.transpose(image, (1, 2, 0)), brightness_factor), (2, 0, 1))
        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            image = np.transpose(adjust_saturation(np.transpose(image, (1, 2, 0)), saturation_factor), (2, 0, 1))
        return image

        
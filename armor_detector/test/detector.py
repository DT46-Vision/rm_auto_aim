import cv2
import numpy as np

# 定义常量
RED = 'red'
BLUE = 'blue'
ARMOR_TYPE_STR = ['INVALID', 'SMALL', 'LARGE']

class Light:
    def __init__(self, rect):
        self.bounding_rect = rect
        self.top = (rect[0][0], rect[0][1])  # 记录灯的上点
        self.bottom = (rect[1][0], rect[1][1])  # 记录灯的下点
        self.center = np.array([(rect[0][0] + rect[1][0]) / 2, (rect[0][1] + rect[1][1]) / 2])
        self.width = rect[1][0] - rect[0][0]
        self.length = rect[1][1] - rect[0][1]
        self.color = None  # 灯的颜色

class Armor:
    def __init__(self, light1, light2):
        self.left_light = light1
        self.right_light = light2
        self.type = None  # 装甲板类型
        self.classfication_result = ""  # 分类结果
        self.number_img = None  # 数字图像

class Detector:
    def __init__(self, bin_thres, color, light_params, armor_params):
        self.binary_thres = bin_thres
        self.detect_color = color
        self.l = light_params
        self.a = armor_params
        self.armors_ = []
        self.lights_ = []

    def detect(self, input_img):
        binary_img = self.preprocess_image(input_img)  # 预处理图像
        self.lights_ = self.find_lights(input_img, binary_img)  # 查找灯源
        self.armors_ = self.match_lights(self.lights_)  # 匹配灯源

        if self.armors_:
            # 这里可以添加数字提取和分类的逻辑
            pass

        return self.armors_

    def preprocess_image(self, rgb_img):
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        _, binary_img = cv2.threshold(gray_img, self.binary_thres, 255, cv2.THRESH_BINARY)  # 二值化
        return binary_img

    def find_lights(self, rgb_img, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
        lights = []

        for contour in contours:
            if len(contour) < 5:
                continue

            r_rect = cv2.minAreaRect(contour)  # 获取最小外接矩形
            light = Light(r_rect)

            if self.is_light(light):
                lights.append(light)

        return lights

    def is_light(self, light):
        ratio = light.width / light.length  # 计算长宽比
        ratio_ok = self.l['min_ratio'] < ratio < self.l['max_ratio']
        angle_ok = light.tilt_angle < self.l['max_angle']  # 可根据具体实现添加角度计算

        return ratio_ok and angle_ok

    def match_lights(self, lights):
        armors = []
        
        for i, light_1 in enumerate(lights):
            for light_2 in lights[i+1:]:
                if light_1.color != self.detect_color or light_2.color != self.detect_color:
                    continue

                if self.contain_light(light_1, light_2, lights):
                    continue

                armor_type = self.is_armor(light_1, light_2)
                if armor_type != 'INVALID':
                    armor = Armor(light_1, light_2)
                    armor.type = armor_type
                    armors.append(armor)

        return armors

    def contain_light(self, light1, light2, lights):
        points = [light1.top, light1.bottom, light2.top, light2.bottom]
        bounding_rect = cv2.boundingRect(np.array(points))  # 计算边界框

        for test_light in lights:
            if test_light.center == light1.center or test_light.center == light2.center:
                continue

            if (bounding_rect[0] <= test_light.top[0] <= bounding_rect[0] + bounding_rect[2] and
                bounding_rect[1] <= test_light.top[1] <= bounding_rect[1] + bounding_rect[3]):
                return True

        return False

    def is_armor(self, light1, light2):
        light_length_ratio = min(light1.length, light2.length) / max(light1.length, light2.length)
        light_ratio_ok = light_length_ratio > self.a['min_light_ratio']

        avg_light_length = (light1.length + light2.length) / 2
        center_distance = np.linalg.norm(light1.center - light2.center) / avg_light_length
        center_distance_ok = (self.a['min_small_center_distance'] <= center_distance < self.a['max_small_center_distance'] or
                              self.a['min_large_center_distance'] <= center_distance < self.a['max_large_center_distance'])

        angle = np.degrees(np.arctan2(light1.center[1] - light2.center[1], light1.center[0] - light2.center[0]))
        angle_ok = abs(angle) < self.a['max_angle']

        is_armor = light_ratio_ok and center_distance_ok and angle_ok

        return 'LARGE' if is_armor and center_distance > self.a['min_large_center_distance'] else 'SMALL' if is_armor else 'INVALID'

    def draw_results(self, img):
        for light in self.lights_:
            cv2.circle(img, light.top, 3, (255, 255, 255), 1)
            cv2.circle(img, light.bottom, 3, (255, 255, 255), 1)
            line_color = (255, 255, 0) if light.color == RED else (255, 0, 255)
            cv2.line(img, light.top, light.bottom, line_color, 1)

        for armor in self.armors_:
            cv2.line(img, armor.left_light.top, armor.right_light.bottom, (0, 255, 0), 2)
            cv2.line(img, armor.left_light.bottom, armor.right_light.top, (0, 255, 0), 2)

        for armor in self.armors_:
            cv2.putText(img, armor.classfication_result, armor.left_light.top, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# 使用示例
# light_params 和 armor_params 应根据实际需求设置
light_params = {'min_ratio': 0.5, 'max_ratio': 2.0, 'max_angle': 15}
armor_params = {'min_light_ratio': 0.5, 'min_small_center_distance': 30, 'max_small_center_distance': 100,
                'min_large_center_distance': 100, 'max_large_center_distance': 300, 'max_angle': 15}

detector = Detector(bin_thres=128, color=RED, light_params=light_params, armor_params=armor_params)

# 读取图像并进行检测
input_image = cv2.imread('image_path.jpg')
detected_armors = detector.detect(input_image)

# 绘制检测结果
detector.draw_results(input_image)
cv2.imshow('Detection Results', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
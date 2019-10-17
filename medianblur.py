import cv2
import numpy as np


def medianBlur(img, kernel, padding_way='ZERO'):
    # kernel的值为奇数
    if kernel % 2 == 0 or kernel is 1:
        print('kernel must be odd')
        return None

    padding = kernel // 2        # 计算padding的大小
    channel = len(img.shape)       # 获取图片的通道数
    height, width = img.shape[:2]  # 获取图片的大小

    if channel == 3:  # 三通道拆成单通道
        matrix_3 = np.zeros_like(img)
        for l in range(matrix_3.shape[2]):
            matrix_3[:, :, l] = medianBlur(img[:, :, l], kernel, padding_way)
        return matrix_3
    elif channel == 2:  # 单通道
        matrix = np.zeros((height + padding * 2, width + padding * 2), dtype=img.dtype)
        matrix[padding:-padding, padding:-padding] = img

        #padding way
        if padding_way is 'ZERO':
            pass
        elif padding_way is 'REPLICA':
            for r in range(padding):
                matrix[r, padding:-padding] = img[0, :]
                matrix[-(1 + r), padding:-padding] = img[-1, :]
                matrix[padding:-padding, r] = img[:, 0]
                matrix[padding:-padding, -(1 + r)] = img[:, -1]
        else:
            print('padding_way error need ZERO or REPLICA')
            return None

        # 创建输出矩阵
        mat_median = np.zeros((height, width), dtype=img.dtype)
        # 遍历矩阵的每个像素
        for x in range(height):
            for y in range(width):
                # kernel转化成队并列
                line = matrix[x:x + kernel, y:y + kernel].flatten()
                line = np.sort(line)    #排序
                # 取中间值赋值
                mat_median[x, y] = line[(kernel * kernel) // 2]
        return mat_median
    else:
        print('image layers error')
        return None


def main():
    img = cv2.imread("lenna.png", 1)
    img_median = medianBlur(img, 5, padding_way='REPLICA')  #输入图片和卷积核尺寸
    cv2.imshow('lenna', img)
    cv2.imshow('medianBlur_lenna', img_median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

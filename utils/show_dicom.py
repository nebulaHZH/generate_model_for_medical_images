import pydicom
import matplotlib.pyplot as plt


# 加载 DICOM 文件
ds = pydicom.dcmread('example/image_0')

# 显示 DICOM 文件
plt.imshow(ds.pixel_array, cmap='gray')
plt.show()
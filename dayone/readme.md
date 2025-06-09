
# Sentinel-2 RGB图像处理工具文档

## 功能概述
该工具用于处理Sentinel-2卫星的多波段数据，将指定的RGB波段组合并压缩为标准的8位RGB图像。

## 主要功能
- 读取Sentinel-2的多个波段TIFF文件
- 将原始反射率值(0-10000)压缩到0-255范围
- 可选的百分比拉伸和伽马校正
- 支持自定义RGB波段组合
- 可保存为图像文件或直接显示

## 安装依赖
```bash
pip install numpy pillow rasterio matplotlib
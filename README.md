# 使用Paddle复现NGM算法

下载预训练模型参数，并且放于pretrained目录下，参数下载链接为：

| [![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAyJpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RSZWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZVJlZiMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENTNiAoV2luZG93cykiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6Q0YyQjQ5QzRBM0NCMTFFOTkxQjVGMjY0QTIzMUIzMzMiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6Q0YyQjQ5QzVBM0NCMTFFOTkxQjVGMjY0QTIzMUIzMzMiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDpDRjJCNDlDMkEzQ0IxMUU5OTFCNUYyNjRBMjMxQjMzMyIgc3RSZWY6ZG9jdW1lbnRJRD0ieG1wLmRpZDpDRjJCNDlDM0EzQ0IxMUU5OTFCNUYyNjRBMjMxQjMzMyIvPiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/PmQOz60AAAFYSURBVHja7JrNSsNAEMdnt7uSxoNaDwURPemhHgXfpILP4qkHX8V36cGrJ0FUqAdBjKFtyDa77gaKCCl4Gd2E/8BklhCy89v5OiTCOUddEPEyvfoLkjOv9xwvPry4ra0Kl4PjczaC/GNGefY69ssJ50lJ7lD0twfBXHLvE0A+na3YNlA6CTryyxE3iCIhWh+VAJIKwZth/XQvmDF71+Is9rVk7880z9/4UtjrwjnLHpWdwVGtHDJ7uqtTy2waiuF+WwamarppTEXLpSGz8k1A+kds4fuB+HdntZKUJJq07v0OpCgM7Z5cR3ny2cNNI0hjYZiVjTaFNvkmqSMCkBi7Vs+UC5Lyu4BsVVA5f4zS4eCbKZsnu9VbqWhzNEIg6va7Pzz9EZG2yXqyo9gBAhCAAAQgAAEIQAACEIAABCAAAQhAAAIQgAAEIFFK/Vkh2NaDdOXHsy8BBgC3bWGcxvvnYgAAAABJRU5ErkJggg==)](https://jbox.sjtu.edu.cn/l/C1cJZT) | 分享内容: [             params           ](https://jbox.sjtu.edu.cn/l/C1cJZT)                链接地址:https://jbox.sjtu.edu.cn/l/C1cJZT |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
|                                                              |                                                              |      |
| 来自于: yaojin                                               |                                                              |      |



在主目录下运行以下命令可分别测试voc和willow测试集结果。

```python
python eval_pdl2.py --cfg='experiments/vgg16_ngm_voc.yaml'
python eval_pdl2.py --cfg='experiments/vgg16_ngm_willow.yaml'
```


# Reproducing NGM algorithm with paddle

From [jbox](https://jbox.sjtu.edu.cn/l/C1cJZT) download pretraining model parameters, and put it in the "pretrained" directory.

Run the following command in the home directory to test the results of VOC and willow test dataset respectively.

```shell
python eval_pdl2.py --cfg='experiments/vgg16_ngm_voc.yaml'
python eval_pdl2.py --cfg='experiments/vgg16_ngm_willow.yaml'
```

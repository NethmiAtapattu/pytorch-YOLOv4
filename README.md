# Pytorch-YOLOv4

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/


- [x] Inference
- [x] Train
    - [x] Mocaic

```
├── README.md
├── demo.py               demo to run pytorch --> tool/darknet2pytorch
├── models.py             model for pytorch
├── train.py              train models.py
├── cfg.py                cfg.py for train
├── cfg                   cfg --> darknet2pytorch
├── data            
├── weight                --> darknet2pytorch
├── tool
│   ├── coco_annotation.py       
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

![image](https://user-gold-cdn.xitu.io/2020/4/26/171b5a6c8b3bd513?w=768&h=576&f=jpeg&s=78882)

# 0. Weights Download

## 0.1 darknet
- baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
- google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

## 0.2 pytorch
you can use darknet2pytorch to convert it yourself, or download my converted model.

- baidu
    - yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) 
    - yolov4.conv.137.pth(https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA Extraction code:kcel)
- google
    - yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    - yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

# 1. Train

[use yolov4 to train your own data](Use_yolov4_to_train_your_own_data.md)

1. Download weight
2. Transform data

    For coco dataset,you can use tool/coco_annotation.py.
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
3. Train

    you can set parameters in cfg.py.
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```

# 2. Inference

## 2.1 Performance on MS COCO dataset (using pretrained DarknetWeights from <https://github.com/AlexeyAB/darknet>)

**ONNX and TensorRT models are converted from Pytorch (TianXiaomo): Pytorch->ONNX->TensorRT.**
See following sections for more details of conversions.

- val2017 dataset (input size: 416x416)

| Model type          | AP          | AP50        | AP75        |  APS        | APM         | APL         |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| DarkNet (YOLOv4 paper)|     0.471 |       0.710 |       0.510 |       0.278 |       0.525 |       0.636 |
| Pytorch (TianXiaomo)|       0.466 |       0.704 |       0.505 |       0.267 |       0.524 |       0.629 |
| TensorRT FP32 + BatchedNMSPlugin | 0.472| 0.708 |       0.511 |       0.273 |       0.530 |       0.637 |
| TensorRT FP16 + BatchedNMSPlugin | 0.472| 0.708 |       0.511 |       0.273 |       0.530 |       0.636 |

- testdev2017 dataset (input size: 416x416)

| Model type          | AP          | AP50        | AP75        |  APS        | APM         | APL         |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| DarkNet (YOLOv4 paper)|     0.412 |       0.628 |       0.443 |       0.204 |       0.444 |       0.560 |
| Pytorch (TianXiaomo)|       0.404 |       0.615 |       0.436 |       0.196 |       0.438 |       0.552 |
| TensorRT FP32 + BatchedNMSPlugin | 0.412| 0.625 |       0.445 |       0.200 |       0.446 |       0.564 |
| TensorRT FP16 + BatchedNMSPlugin | 0.412| 0.625 |       0.445 |       0.200 |       0.446 |       0.563 |


## 2.4 Inference output
 Rating specific for Substance Detection

## 3 Run the demo

```sh
python demo_trt.py <tensorRT_engine_file> <input_image> <input_H> <input_W>
```

- This demo here only works when batchSize is dynamic (1 should be within dynamic range) or batchSize=1, but you can update this demo a little for other dynamic or static batch sizes.
    
- Note1: input_H and input_W should agree with the input size in the original ONNX file.
    
- Note2: extra NMS operations are needed for the tensorRT output. This demo uses python NMS code from `tool/utils.py`.


 
  
   
Reference:
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3



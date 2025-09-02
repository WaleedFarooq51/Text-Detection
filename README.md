# Arbitrary Shape Text Detection

This repository implements an **arbitrary shape text detection** using **Detectron 2** for instance segmentation and **Mask-RCNN** as a backbone for extracting text features. 

---

## 📁 Dataset

- The dataset includes the images of arbitrary-shaped text with varied backgrounds.

- The data is annotated using `LabelMe` tool. Annotations are of bounding boxes with polygonal shapes, and labels are stored in `.json` format.

- The dataset can be downloaded using the [link]().

## 📊 Results

Example results of our trained model on our test dataset.

![Sample Detection](./results/img1.png) ![Sample Detection](./results/img2.png) ![Sample Detection](./results/img3.png)

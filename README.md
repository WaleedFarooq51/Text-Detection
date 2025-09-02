# Arbitrary Shape Text Detection

This repository implements an **arbitrary shape text detection** using **Detectron 2** for instance segmentation and **Mask-RCNN** as a backbone for extracting text features. 

---

## 📁 Dataset

- The dataset includes the images of arbitrary-shaped text with varied backgrounds.

- The data is annotated using `LabelMe` tool. Annotations are of bounding boxes with polygonal shapes, and labels are stored in `.json` format.

- The dataset can be downloaded using the [link]().

## 📊 Results

Example results of our trained model on our test dataset.

![Sample Detection](./results/BEAR.jpg) ![Sample Detection](./results/BUS.jpg) ![Sample Detection](./results/FUZZ.jpg)

## Installation

```bash
1. Clone the repo:
   git clone <repo-url>
   cd <repo-name>

2. Install required libraries:
   pip install requirements.txt

3. Open text_detection.ipynb notebook and follow the installation instructions for the required libraries in it.
```

## Inference


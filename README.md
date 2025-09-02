# Arbitrary Shape Text Detection

This repository implements an **arbitrary shape text detection** using **Detectron 2** for instance segmentation and **Mask-RCNN** as a backbone for extracting text features. 

---

## 📁 Dataset

- The dataset includes the images of arbitrary-shaped text with varied backgrounds.

- The data is annotated using `LabelMe` tool. Annotations are of bounding boxes with polygonal shapes, and labels are stored in `.json` format.

- The dataset can be downloaded using the [link]().

## 📊 Results

Example results of the trained model on test dataset.

![Sample Detection](./results/BEAR.jpg) ![Sample Detection](./results/BUS.jpg) ![Sample Detection](./results/FUZZ.jpg)

## - Installation

```bash
1. Clone the repo:
   git clone <repo-url>
   cd <repo-name>

2. Install required libraries:
   `pip install requirements.txt`

3. Open text_detection.ipynb notebook and follow the installation instructions for the required libraries in it.
```

## - Inference

- If you want to carry out inference using my trained model available [here](), execute `detection_inference.py` in the `inference` folder using following command:

  `python inference/detection_inference.py \--config-file configs/ocr/config.yaml\--testdata test_images\--weights out_dir/trained_model/model_final.pth

- Store the trained model weights in the out_dir folder and set the path in the above command.

- Similar command have to be used if you want to carryout inferecne using your own trained model weights.


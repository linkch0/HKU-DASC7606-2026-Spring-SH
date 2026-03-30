# HKU DASC7606 2026 Spring SH Assignment

In this assignment, you are required to implement the **CLIP (Contrastive Language-Image Pretraining)** model. The CLIP model will be trained on the subset of ImageNet-1K dataset and evaluated on the given test dataset in a **zero-shot** manner. GitHub Link: <https://github.com/linkch0/HKU-DASC7606-2026-Spring-SH>

The following resources will be provided on Moodle:

- [FAQ & Discussion Online Documentation](https://moodle.hku.hk/mod/url/view.php?id=4090272)
- [Test Dataset download link](https://moodle.hku.hk/mod/url/view.php?id=4090275)


**Authors (TAs): Link Chen**  

This codebase is only for HKU DASC7606 2026 Shanghai course. Please **don't upload your answers or this codebase** to **any** public platforms (e.g., GitHub) before permitted. All rights reserved.

---

## 1. Introduction

### 1.1 What is CLIP and Contrastive Learning?

CLIP (Contrastive Language-Image Pretraining), developed by OpenAI, is a neural network model trained on a vast collection of image-text pairs. Unlike traditional vision models that are trained on a fixed set of predefined categories, CLIP learns a joint embedding space for images and text, enabling it to match images with their textual descriptions. This is achieved through **contrastive learning**, where the model is trained to bring matching image-text pairs close together in the embedding space while pushing non-matching pairs apart.

### 1.2 What is Zero-Shot Classification?

A remarkable capability of CLIP is **zero-shot classification**: the ability to classify images into categories it has never explicitly been trained on. Given a set of textual descriptions of categories (e.g., *"a photo of a cat"*, *"a photo of a dog"*), CLIP can predict which description best matches a given image — without any task-specific fine-tuning. This assignment focuses on fine-tuning CLIP on the subset of ImageNet-1K dataset and then evaluating its zero-shot transfer performance on held-out test datasets.

### 1.3 What Will You Learn from This Assignment?

- Understand the **CLIP model architecture**, including the image encoder (ResNet50 / ViT-B/16) and the text encoder (BERT based), and how they are jointly trained with a contrastive objective.
- Gain hands-on experience with **contrastive learning**, including how to implement the symmetric cross-entropy loss over a similarity matrix.
- Learn how to apply **zero-shot classification** to image recognition tasks using language prompts.
- Develop a complete deep learning pipeline with [**PyTorch**](https://pytorch.org/), [**transformers**](https://huggingface.co/transformers/) covering model design, training, hyperparameter tuning, and performance evaluation.
- Gain practical experience of working with large-scale vision-language datasets and pre-trained model weights.

### 1.4 What Should You Prepare?

- Master the basic use of Python and PyTorch.
- Be familiar with standard neural network architectures such as [**ResNet**](https://arxiv.org/abs/1512.03385) and [**Vision Transformer (ViT)**](https://arxiv.org/abs/2010.11929).
- Refer to the original [**CLIP paper**](https://arxiv.org/abs/2103.00020) by Radford et al. to understand the model design and training objective.
- Great [**Video**](https://www.bilibili.com/video/BV1SL4y1s7LQ) Explanation of CLIP Paper

*(You can learn about them step by step in the process of completing this assignment.)*

---

## 2. Setup

Students may complete the assignment via two primary environments: remotely on the HKU CS GPU Farm (recommended) or locally on their personal computer.

### 2.1 Working Remotely on HKU CS GPU Farm (Recommended)

Training the model requires the usage of GPU resources. You can apply for access to the HKU CS GPU Farm (GPU Farm for Teaching, formerly Phase 2).

1. **Application System**: <https://intranet.cs.hku.hk/gpufarm_acct_cas/>
2. **Quick Start Guide**: <https://cs.hku.hk/gpu-farm/quickstart/>
3. **Login** with your username and password (or SSH key), e.g.:
    - Gateway nodes for GPU Farm Phase 2: `gpu2gate1.cs.hku.hk` or `gpu2gate2.cs.hku.hk`
    - `ssh <your_portal_id>@gpu2gate1.cs.hku.hk`
4. **Using GPUs in interactive mode**:
    - Run a bash shell on a GPU node: `gpu-interactive`
    - Verify that a GPU is allocated: `nvidia-smi`
5. **Avoid Idle Sessions and Use tmux for Unstable Network Connections**:
    - A Quick and Easy Guide to tmux: <https://hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/>
    - Start a tmux session: `tmux`
    - Resume the session: `tmux attach`

### 2.2 Working Locally

If you have GPU resources on your personal computer, you may work locally by installing the GPU drivers, CUDA, cuDNN, and PyTorch. Completing the assignment without a GPU is feasible, but model training and inference will be significantly slower. Therefore, having a GPU offers a substantial advantage throughout the assignment.

### 2.3 Creating Python Environments

**Installing Miniconda**: Miniconda is a free, miniature installation of Anaconda that includes conda and Python. Install: <https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2>

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

**Virtual Environment**: Create and activate a Conda virtual environment:

```shell
conda create -n clip_env python=3.10
conda activate clip_env
```

**Install Required Libraries**: Install PyTorch following the [official guidelines](https://pytorch.org/get-started/locally/), then install the project dependencies:

```shell
pip install -r requirements.txt
```

**JupyterLab** (for running tutorial notebooks on the GPU Farm):

```shell
pip install jupyterlab
```

Running JupyterLab without starting a web browser:

1. Login to a GPU compute node from a gateway node: `gpu-interactive`
2. Find the IP address of the GPU compute node: `hostname -I`
3. Start JupyterLab: `jupyter lab --no-browser`
4. On your local PC, start another terminal and run SSH with port forwarding:
    ```shell
    ssh -L 8888:localhost:8888 -J <username>@gpu2gate1.cs.hku.hk <username>@10.XXX.XXX.XXX
    ```
5. Open a web browser and navigate to the URL shown in step 3.

---

## 3. Working on the Assignment

### 3.1 Dataset and Model Preparation

**Download ImageNet Dataset**: Run `hf auth login` and input your Hugging Face token, then run:

```shell
python data/download.py
```

This script will take around 2 hours to download and save ~250,000 images. Make sure you run it before you start the assignment. The downloaded ImageNet dataset will be saved in the `data/imagenet/` folder.

**Download Pretrained Model Weights**: Run the tutorial notebooks provided in the `tutorial/` folder. These notebooks are provided for you to understand the CLIP model architecture and the datasets. You can run them via JupyterLab on the GPU Farm.

1. `tutorial/dataset.ipynb`: Introduction to the ImageNet datasets, including how to load and visualize them.
2. `tutorial/model.ipynb`: Introduction to the CLIP model architecture, including the image encoder, text encoder, and the contrastive learning objective.
3. `tutorial/demo.ipynb`: A demo of how to use the CLIP model for zero-shot classification on a batch of images.

### 3.2 Code Structure

The codebase is organized as follows:

```
├── model/
│   ├── image_encoder.py    # ResNet50 and ViT image encoders
│   ├── text_encoder.py     # RoBERTa based text encoder
│   └── clip.py             # Main CLIP model
├── data/
│   ├── download.py         # Script to download ImageNet dataset
│   └── dataset.py          # Data loading and preprocessing
├── tutorial/
│   ├── dataset.ipynb       # Dataset introduction and visualization
│   ├── model.ipynb         # CLIP model architecture walkthrough
│   └── demo.ipynb          # Zero-shot classification demo
├── config.py               # Configuration and hyperparameters
├── utils.py                # Training and evaluation utilities
├── train.ipynb             # Fine-tuning CLIP on ImageNet
├── eval.ipynb              # Zero-shot evaluation on test datasets
├── predict.py              # Script for making predictions for submission
└── requirements.txt        # Project dependencies
```

### 3.3 Assignment Tasks

#### Task 1: Environment Setup

Apply for an HKU GPU Farm account (if you don't have one) and setup the Python environment following the instructions in Section 2.

#### Task 2: Data and Model Download

Download the pretrained model weights (ResNet50, ViT-B/16) and the ImageNet-1K training dataset as described in Section 3.1.

#### Task 3: Implement the CLIP Model

Fill in all the `TODO` sections in the provided code template. The key files to complete are:

- `model/image_encoder.py`: Implement the image encoding pipeline.
- `model/text_encoder.py`: Implement the text encoding pipeline.
- `model/clip.py`: Implement the full CLIP forward pass and the contrastive loss function.
- `utils.py`: Implement training utilities (e.g., data colloator, metrics computation, top-k evaluation).

#### Task 4: Fine-tune the CLIP Model

Fine-tune the CLIP model on the ImageNet-1K training dataset using `train.ipynb`. You may tune hyperparameters (e.g., learning rate, batch size, number of epochs in `config.py`) to improve performance. Your model should converge to a reasonable accuracy on the validation set.

#### Task 5: Zero-Shot Evaluation

Evaluate the zero-shot performance of your fine-tuned CLIP model on the given test datasets using `eval.ipynb`. Report the top-k accuracies. You are encouraged to experiment with different text prompt templates to improve zero-shot accuracy.

Change the best model checkpoint path in `config.py` then run `predict.py` to generate the `prediction.json` file for submission. The `prediction.json` file should follow the format:

```json
[
  {
    "filename": "0001.jpg",
    "label_id": n,
    "label_name": "xxxx"
  },
  ...
]
```

#### Task 6: Write a Report

Write a technical report (up to 4 pages in PDF) describing your experimental results and analysis. Your report should cover:

- **Method**: Describe your implementation and any modifications you made to the baseline.
- **Experiments**: Report your fine-tuning setup (hyperparameters, training curves) and zero-shot evaluation results.
- **Analysis**: Analyze the effect of different design choices (e.g., prompt engineering, encoder backbone, learning rate). Provide visualizations such as the training loss curve. Discuss the failure cases of zero-shot classification and analyze the reasons behind them.

*Spontaneously explore advanced techniques, such as prompt ensembling or using stronger backbone architectures.*

## 4. Submission

### 4.1 Submission File Structure

You are required to submit a ZIP file containing your code and report. The ZIP file should be named `student_id.zip` (replace `student_id` with your actual student ID) and organized as follows:

```
└── student_id/
    ├── src/                # Your completed source code
    ├── prediction.json      # Your prediction results for the test dataset (in JSON format)
    └── report.pdf          # Experiment report (maximum of 4-page)
```

The `src` folder should be same structure as the original codebase, including `*.py, *.ipynb` files. Do not include the downloaded datasets or model weights in your submission. The `prediction.json` file should be generated by running `predict.py` on the test dataset.

The ZIP file should be less than 100MB and submitted via Moodle before the deadline.

### 4.2 Submission Deadline

The submission deadline is **April 26 Sunday**.

**Late submission policy**:

- 10% deduction for late assignments submitted within 1 day late.
- 20% deduction for late assignments submitted within 2 days late.
- 50% deduction for late assignments submitted within 7 days late.
- 100% deduction for late assignments submitted after 7 days late.

---

## 5. Marking Scheme

The final score of the assignment is composed of three parts: (1) the completed codebase (20%); (2) the zero-shot classification performance (40%); (3) the experiment analysis report (40%).

1. **Completed Codebase (20%)**: Marks will be given mainly based on the correctness and completeness of your implementation. TAs will re-run your code to verify the results.

2. **Model Performance (40%)**: Marks will be given based on the zero-shot top-1, top-5, top-10 accuracy of your fine-tuned model on the held-out test set.

    - Top-1 Accuracy (25%): Correct label is ranked #1

        | Accuracy | Marks |
        | -------- | ----- |
        | < 15%    | 0     |
        | 15–39%   | 15    |
        | 40–49%   | 18    |
        | 50–59%   | 21    |
        | ≥ 60%    | 25    |

    - Top-5 Accuracy (15%): Correct label appears in top 5 predictions

        | Accuracy | Marks |
        | -------- | ----- |
        | < 30%    | 0     |
        | 30–49%   | 10    |
        | 50–69%   | 12    |
        | 70–89%   | 14    |
        | ≥ 90%    | 15    |

    - Top-10 Accuracy (10%): Correct label appears in top 10 predictions

        | Accuracy | Marks |
        | -------- | ----- |
        | < 40%    | 0     |
        | 50–69%   | 6     |
        | 70–89%   | 8     |
        | ≥ 90%    | 10    |

3. **Experiment Analysis Report (40%)**: Marks will be given mainly based on the richness of experiments and analysis.
    - Rich experiments + detailed analysis: 90%–100% mark of this part.
    - Reasonable number of experiments + analysis: 70%–80% mark of this part.
    - Basic analysis: 50%–60% mark of this part.
    - Not sufficient analysis: lower than 50%.

4. $\text{Final mark} = \text{Codebase} + \text{Performance} + \text{Report}$


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WG_g337P)
# E4040 2025 Fall Project
## TODO: Your Project Title

Repository for E4040 2025 Fall Project
  - Distributed as Github repository and shared via Github Classroom
  - Contains only `README.md` file


# Topic: Residual Attention Network Reproduction (CIFAR-10)

This repository contains our code of reproduction and analysis of the **Residual Attention Network for Image Classification** proposed by Wang et al. (2017). The goal of this project is to study the effectiveness of residual attention learning, analyze the impact of network depth under practical training constraints, and evaluate model robustness under noisy supervision.

All experiments are conducted on the CIFAR-10 dataset using TensorFlow.


## Project Objectives

The project is organized around three main objectives:

1. **Effect of Residual Attention Learning**  
   Compare naive attention learning (NAL) and attention residual learning (ARL) using different network depths to examine the impact of residual connections in attention modules.

2. **Effect of Network Depth and Model Complexity**  
   Analyze Attention-56, Attention-92, Attention-128, and Attention-164 under both the original architecture and a simplified configuration, and identify the most effective model under constrained training settings.

3. **Robustness to Noisy Labels**  
   Evaluate the robustness of Attention92 Using ResNet92 as baseline model under varying levels of label noise, following a uniform noise model.



## Models Implemented

We implement the following Residual Attention Network variants:

- Attention-56  
- Attention-92  
- Attention-128  
- Attention-164  

For each model, two architectural configurations are considered:

- **Original version**:  
  The number of attention modules per stage increases with network depth, following the design philosophy of the original paper.
  1 attention module in each stage in Attention56(total 1 x 3 = 3)
  2 attention module in each stage in Attention96(total 2 x 3 = 6)
  3 attention module in each stage in Attention128(total 3 x 3 = 9)
  4 attention module in each stage in Attention164(total 4 x 3= 12)

- **Simple version**:  
  Each stage contains exactly one attention module, while preserving the overall depth and trunk structure of the network.

In addition, **ResNet92** is implemented as a baseline for robustness experiments.



## Dataset

All experiments are conducted on **CIFAR-10**, which contains 50,000 training images and 10,000 test images across 10 classes. The test set remains clean for all experiments.



## Training Setup

- Framework: TensorFlow  
- Optimizer: Adam  
- Training epochs: 50  
- Batch size: 128  
- Hardware: NVIDIA T4 GPU  

Due to computational constraints, we adopt a fixed training budget across all experiments to ensure fair comparison between models.



## Robustness Evaluation

To evaluate robustness, we introduce **label noise** to the training set while keeping the test set unchanged. Label noise is injected using a **uniform noise model**, which can be described by a label transition matrix \(Q\), where a fixed fraction of labels is randomly flipped to other classes.

We consider four noise levels:

- 10%  
- 30%  
- 50%  
- 70%  
!!Attention: But only 10% and 30% noise level were completely implemented cause the robustness process is too time-consuming.

Robustness experiments compare **Attention92** and **ResNet92**, as these models achieve the highest accuracy among their respective architectures under clean labels.



## Experimental Results Summary

Key findings from our experiments include:

- Residual attention learning generally improves model performance, but it marginal reversed on Attention168.
- Increasing network depth does not guarantee better performance under our limited training budgets.
- The simplified attention configuration consistently reduces error compared to the original design.
- Attention92 provides the best balance between performance, stability, and computational efficiency.
- Under noisy dispose, Attention92 demonstrates stronger robustness than ResNet92, especially at higher noise levels.


## Future Work
The most essential shortage of our study is the computational limitations. Most of our inconsistent results with the origin paper can be solved if we have stronger GPU to train the model. Also there is no clarification of the number of attention modules in each attention-based model in the original paper so we implemented following our own ideas, which might not be appropriate. We did not tune hyperparameters very carefully since computational limitation. Further hyperparameter tuning might helps improve the results a lot.



## Detailed instructions how to submit this project:
1. The project will be distributed as a Github classroom assignment - as a special repository accessed through a link
2. A student's copy of the assignment gets created automatically with a special name
3. **Students must rename the repository per the instructions below**
5. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud
6. If some model is too large to be uploaded to Github - 1) create google (liondrive) directory; 2) upload the model and grant access to e4040TAs@columbia.edu; 3) attach the link in the report and this `README.md`
7. Submit the report as a PDF in the root of this Github repository
8. Also submit the report as a PDF in Courseworks
9. All contents must be submitted to Gradescope for final grading

## TODO: (Re)naming of a project repository shared by multiple students
Students must use a 4-letter groupID, the same one that was chosen in the class spreadsheet in Google Drive: 
* Template: e4040-2025Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2025Fall-Project-MEME-zz9999-aa9999-aa0000.

# Organization of this directory
To be populated by students, as shown in previous assignments.

TODO: Create a directory/file tree
```
./e4040-fall2025-project-swye-xs2543-jw4588-my2903
├── 4040_final_project_report_swye.pdf
├── Plot.ipynb
├── README .md
├── README.md
├── checkpoints
├── demo_notebooks
│   ├── demo_code_1.ipynb
│   ├── demo_code_2.ipynb
│   ├── demo_code_3.ipynb
│   └── demo_code_4.ipynb
├── figures
│   ├── jw4588_gcp_work_screenshot_1.png
│   ├── jw4588_gcp_work_screenshot_2.png
│   ├── jw4588_gcp_work_screenshot_3.png
│   ├── my2903_gcp_work_screenshot_1.png
│   ├── my2903_gcp_work_screenshot_2.png
│   ├── my2903_gcp_work_screenshot_3.png
│   ├── xs2543_gcp_work_screenshot_1.png
│   ├── xs2543_gcp_work_screenshot_2.png
│   └── xs2543_gcp_work_screenshot_3.png
├── models
│   ├── __init__.py
│   ├── __init__tf.py
│   ├── __pycache__
│   │   ├── __init__.cpython-36.pyc
│   │   ├── attention128_tf.cpython-36.pyc
│   │   ├── attention164_tf.cpython-36.pyc
│   │   ├── attention56.cpython-36.pyc
│   │   ├── attention56_tf.cpython-36.pyc
│   │   ├── attention92.cpython-36.pyc
│   │   ├── attention92_tf.cpython-36.pyc
│   │   └── layers_tf.cpython-36.pyc
│   ├── attention128_simple.py
│   ├── attention128_tf.py
│   ├── attention164_simple.py
│   ├── attention164_tf.py
│   ├── attention56.py
│   ├── attention56_simple.py
│   ├── attention56_tf.py
│   ├── attention92_simple.py
│   ├── attention92_tf.py
│   ├── layers_tf.py
│   ├── resnet128_tf.py
│   ├── resnet164_tf.py
│   └── resnet92_tf.py
├── paper figures
│   ├── Attention-128_164_architecture.png
│   ├── Attention-56_92_architecture.png
│   ├── mul 56 arl vs nal.png
│   ├── mul ARL.png
│   ├── mul NAL.png
│   ├── one 56 arl vs nal.png
│   ├── one ARL.png
│   └── one NAL.png
├── parameter.py
├── run_all_models.py
├── train_cifar10_robustness.py
├── train_cifar_new 12.14.py
└── train_imagenet.py

6 directories, 53 files

```

# Sudoku solver with AI and Computer Vision

**Can Computer vision and Deep Learning be used to solve Sudoku puzzle?**

I developed an automatic sudoku puzzle solver using computer vision and deep learning. Computer vision is used to extract the Sudoku grid from the image and identify the individual cells of the puzzle, while the deep learning model is used to recognize the number. An algorithm is then used to solve the puzzle. 

The solver can be used to solve a Sudoku puzzle from a single image or in real time from a camera.


## Installation

1. Clone repo 

```bash
git clone https://github.com/CodeWithInnovation/SudokuSolver-AI-ComputerVision 

```
2. Install the dependency using the requirements.txt file in a Python>=3.8.0 environment
```bash
pip install -r requirements.txt  

```
## Train digit classifier
```bash
python train_cnn.py --output model.h5
```
You will get the digit classifier or you can use our pretrained model on the digit_classifier directory

## Solving sudoku puzzle from image
```bash
python sudoku.py --image  examples/sudoku_1.jpg --model digit_classifier/model.h5
```

## Real time sudoku solving
```bash

```
## Example 


## Demo


# Homework 4 - A vision language model for tux

In this homework, we will train (fine-tune) a vision-language model on the SuperTuxKart data [here](https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip).
We will focus on the most important aspect of the VLM pipeline: The data-pipeline.
Model and training algorithms are fun, but what truely makes models work is a good data pipeline.
We will use vision labels of the SuperTuxKart dataset to produce question/answer labels for the same set of images.

The starter code contains a minimal training script and dataset.

- `data.py` load a dataset of *images*, *questions*, and *answers* specified in a json file. See `data/train_demo/balanced_qa_pairs.json` for an example.
- `base_vlm.py` sets of a VLM model that can both train and evaluate on the above training data
- `finetune.py` fine-tunes a VLM on a specific dataset

To get started, familiarize yourself with the starter code, download and unzip the data.

```bash
wget https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip -O supertux_data.zip
unzip supertux_data.zip
```

Then train a model on the demo data we provided

```bash
python -m homework.finetune demo_train
```

and benchmark this model

```bash
python -m homework.finetune test path/to/your/checkpoint
```

Do not expect the model to perform very well, after all it was trained on only 5 question-answer pairs.
Your task is to massively expand this training set.
The checkpoint path needs to include `adapter_config.json` and `adapter_model.safetensors`.

## Grading

To get 100pts on this assignment you should answer 70% if questions correctly.
The score falls off linearly.
There is 5pts extra credit for submissions reaching 85% accuracy (linearly from 80%).

## Building a VLM data-pipeline

Your main task is to massively expand the dataset for VLM training. Follow the 5 kind of questions given in our demo training set.

`generate_qa.py` provides some initial code to parse and visualize the supertux data.

Run

```bash
python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0
```

The visualize the extracted supertuxkart information and your generated questions.
Finally, write question-answer pairs into a `..._qa_pairs.json` file in `data/train/` and train your model using

```bash
python -m homework.finetune train
```

Do NOT train on the validation data provided.
It will inflate your validation accuracy, and unlikely generalizes to the test set.

## Submission

Once you finished the assignment, create a submission bundle using:

```bash
python3 bundle.py homework [YOUR UT ID]
```

Delete any old checkpoints from your homework directory to keep the model size below 50MB.

Submit the zip file on Canvas. Please note that the maximum file size our grader accepts is **50MB**. Please keep your solution compact.
Please double-check that your zip file was properly created, by grading it again:

```bash
python3 -m grader [YOUR UT ID].zip
```

## Online grader

We will use an automated grader through Canvas to grade all your submissions. There is a soft limit of **5** submissions per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of Python and the grader:

- Please do not use the `exit` or `sys.exit` command, it will likely lead to a crash in the grader
- Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
- Network access is disabled. Please do not try to communicate with the outside world.
- Forking is not allowed!
- `print` or `sys.stdout.write` statements from your code are ignored and not returned.

Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

## Installation

We encourage using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required packages.

```bash
conda create --name advances_in_deeplearning python=3.12 -y
conda activate advances_in_deeplearning
```

First, install [PyTorch](https://pytorch.org/get-started/locally/)

Then install additional dependencies:

```bash
pip install -r requirements.txt
```

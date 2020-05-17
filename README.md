# Federated Learning using PHE
## Problem Statement

To show a secure way of sharing medical data from **n** hospitals to a company **X** who helps these hospitals to use machine learning techniques (trained on combined data) to predict if a given breast cancer patient has benign or malignant cancer. This can be generalized to any scenario where the sensitive data is held by different parties but a machine learning model is required to be trainined on the combined data.

## Solution Proposal

Since the task is a binary classification, we will train a logistic regression classification model on the data of all the hospitals without accessing their databases. The **n** hospitals will have different patient records but with the same features.

Our approach to this problem is to not share data in the true or encrypted format but instead share the derived information - gradients (used for gradient descent) in the encrypted format.

The idea is that each hospital computes its own gradient, encrypts it using the public key and then passes it to another hospital which aggregates this to its own computed encrypted gradient and so on. The last hospital passes the total aggregated encrypted gradient to the company **X** which then decrypts it and sends the total gradients back to all the hospitals. The hospitals update their models using this aggregated gradient from all the hospitals.

This allows all the hospitals to have their models trained on the complete sensitive data without actually sharing any information about their personal data. From the security viewpoint, we consider all parties to be "honest but curious".

## Setup

The code was written and tested using -

-   Python 3.7.3
-   Pip 20.0.2

To setup the project and install its dependencies, you need to run the `requirements.txt`.

```bash
pip install -r requirements.txt
```

## How to run

To test the Federated Learning vs the Local training, go to the `src` folder and run the project. This can be done as

```bash
cd src/
python federated_learning_HE.py
```

You can experiment with it using the following flags :

-   dataset choice between grad_school and breast_cancer (`--dataset`, default = breast_cancer)
-   number of clients (`--n_clients`, default = 3)
-   gradient descent learning rate (`--lr`, default = 0.05)
-   number of iterations (`--n_iter`, default = 15)
-   Pallier key length (`--key_length`, default = 1024)

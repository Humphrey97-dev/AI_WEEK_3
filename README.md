AI Tools Assignment: Mastering the AI Toolkit 🛠️🧠
Overview
This repository contains the work for an AI Tools Assignment focused on demonstrating proficiency in using various AI frameworks, including TensorFlow, PyTorch, Scikit-learn, and spaCy. The assignment consists of three parts: theoretical understanding, practical implementation, and ethical considerations (excluded in this repository). The goal is to showcase a deep understanding of AI tools and their real-world applications in a collaborative group setting.

Directory Structure
/AI_Tools_Assignment
    ├── Part_1_Theoretical_Understanding
    ├── Part_2_Practical_Implementation
    │   ├── classical_ml_with_scikit_learn.py
    │   ├── deep_learning_with_tensorflow.py
    │   ├── nlp_with_spacy.py
    ├── README.md
Technologies Used
TensorFlow: Deep learning framework used for building and training the Convolutional Neural Network (CNN) model for MNIST.

Scikit-learn: Machine learning library used for training a Decision Tree Classifier on the Iris dataset.

spaCy: Natural language processing library used for Named Entity Recognition (NER) and sentiment analysis on Amazon product reviews.

Streamlit (for bonus task): A framework used to deploy the MNIST model as a web application for live predictions.

Assignment Breakdown
Part 1: Theoretical Understanding (40%)
In this section, the understanding of AI tools is tested through a series of theoretical questions, including:

TensorFlow vs. PyTorch: Exploring the primary differences between these two deep learning frameworks.

Jupyter Notebooks: Understanding their use cases in AI development, including data analysis and prototyping.

spaCy vs. Basic Python String Operations: Understanding how spaCy enhances NLP tasks beyond what is possible with basic string operations.

Part 2: Practical Implementation (50%)
The practical part of the assignment consists of three tasks:

Classical ML with Scikit-learn (Iris Dataset):

Implement a decision tree classifier to predict the species of flowers in the Iris dataset.

Evaluate the model using metrics like accuracy, precision, and recall.

Deep Learning with TensorFlow (MNIST):

Build a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

Train the model to achieve a test accuracy of over 95% and visualize the results.

NLP with spaCy (Amazon Reviews):

Extract named entities from Amazon product reviews using spaCy’s Named Entity Recognition (NER).

Perform sentiment analysis using a basic rule-based approach to classify reviews as positive or negative.

Part 3: Ethics & Optimization (10%)
This section, which has been excluded from the repository, focuses on ethical considerations in AI development and debugging/optimization tasks. Key points include:

Identifying potential biases in models.

Troubleshooting and optimizing AI models to improve performance and fairness.

Bonus Task (Extra 10%)
Deploy Your Model: Use Streamlit or Flask to create a web interface for the MNIST model, allowing users to interact with the trained model and see live predictions.

How to Run the Code
1. Prerequisites
Ensure that the following libraries are installed:

TensorFlow: For deep learning tasks

Scikit-learn: For classical machine learning tasks

spaCy: For NLP tasks

Matplotlib: For visualizing results

Streamlit (Optional for deployment)

Install the dependencies using the following command:
pip install tensorflow scikit-learn spacy matplotlib streamlit
python -m spacy download en_core_web_sm
2. Running the Scripts
For Part 1: This section is theoretical and does not require code execution.

For Part 2: You can run the scripts for each practical task in your preferred Python environment (e.g., Jupyter Notebook).

Task 1: Run the script for classical ML (classical_ml_with_scikit_learn.py)

Task 2: Run the script for deep learning (deep_learning_with_tensorflow.py)

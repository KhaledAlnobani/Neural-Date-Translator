


#  Neural Date Translator (Seq2Seq with Attention)

An Object-Oriented implementation of a Neural Machine Translation (NMT) model built with TensorFlow/Keras. This model translates highly unstructured, human-readable dates (e.g., "the 29th of August 1958", "03/30/1968", "21 June, 2001") into a standardized machine-readable format (`YYYY-MM-DD`).

##  Project Overview

While date parsing can often be handled by regular expressions (RegEx), highly varied, misspelled, or globally formatted dates quickly break hard-coded rules. This project approaches the problem as a machine translation task using Deep Learning, allowing the model to dynamically "learn" the context of a date string.

### Input vs. Output Examples
| Human-Readable (Input) | Machine-Readable (Target) |
| :--- | :--- |
| `9 may 1998` | `1998-05-09` |
| `10.11.19` | `2019-11-10` |
| `thursday january 26 1995` | `1995-01-26` |

##  Project Structure

The codebase is highly modularized for clean architecture and readability:

* **`NMT.py`**: The core architecture file. Contains the custom `AttentionLayer` class and the `NMTModelBuilder` factory class. Built using the Keras Functional API and Python subclassing for clean, object-oriented weight sharing.
* **`utils.py`**: Contains helper functions for custom activations (like axis-specific softmax) and synthetic data generation using the `Faker` and `Babel` libraries.
* **`main.ipynb`**: The primary Jupyter Notebook. Handles dataset generation, One-Hot Encoding preprocessing, model compilation, training loops, and final inference testing.

## Model Architecture

This model relies on a **Sequence-to-Sequence (Seq2Seq)** architecture enhanced with an **Attention Mechanism**:

1. **The Encoder (Bi-LSTM):** A Bidirectional Long Short-Term Memory network reads the input string character-by-character from both directions. This allows the model to understand the surrounding context of a number (e.g., knowing a "12" is a month because the year "2024" comes directly after it).
2. **The Attention Mechanism (Bahdanau):** Instead of forcing the model to memorize the entire date sequence into a single context vector, the custom attention layer calculates a weighted average of the input features for *every single output step*. It acts as a dynamic "magnifying glass," allowing the decoder to focus only on the specific characters needed right now.
3. **The Decoder (LSTM):** A standard LSTM that takes the dynamically weighted context vector and its own previous hidden state to output the standardized string, character by character.

## Getting Started

### Prerequisites
* Python 3.8+
* TensorFlow 2.x
* NumPy
* Faker
* Babel

### Installation & Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/KhaledAlnobani/Neural-Date-Translator.git](https://github.com/KhaledAlnobani/Neural-Date-Translator.git)
   cd Neural-Date-Translator
   ```
2. Install the required dependencies:
   ```bash
   pip install tensorflow numpy faker babel
   ```
3. Open `main.ipynb` in Jupyter Notebook or VS Code to run the training process and test the translation inference.

##  Acknowledgments
The baseline concepts and utility functions for this project were provided by the excellent *Sequence Models* course by **DeepLearning.ai** (Coursera). The core architecture in `NMT.py` was refactored from a procedural script into a modern, Object-Oriented Keras framework for better modularity and production readiness.

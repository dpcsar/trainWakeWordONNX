# OpenWakeWord Custom Model Training Script

This script trains a custom wake word model for [OpenWakeWord](https://github.com/dscripka/openwakeword). It is adapted from the Colab notebook version to run in WSL or other local environments.

## Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dpcsar/trainWakeWordONNX.git
    cd trainWakeWordONNX
    ```

2.  **Run the setup script:**

    ```bash
    bash train_wake_word.setup.sh
    ```

3.  **Edit the variables:**

    Modify the variables at the top of the [`train_wake_word.py`](train_wake_word.py) file to configure the training process.

4.  **Run the training script:**

    ```bash
    bash train_wake_word.run.sh
    ```

## Configuration

The following parameters can be configured at the beginning of the [`train_wake_word.py`](train_wake_word.py) script:

*   `tf_cpu`: Use TensorFlow CPU (set to `True`) or GPU (set to `False`).
*   `target_words`: A list of target wake words (variants).
*   `model_name`: The name of the model.
*   `number_of_examples`: The number of examples to generate for the wake word.
    *   range: min: 100, max: 50000
*   `number_of_training_steps`: The number of training steps.
    *   range: min: 0, max: 50000
*   `false_activation_penalty`: Controls how strongly false activations are penalized during training.
    *   range: min: 100, max: 5000

## Output

The trained model files are saved in the `my_custom_model` directory. The main model file will be named `<model_name>.onnx`.

## Dependencies

The script has the following dependencies:

*   Python 3.10

Additional requirements are listed in:
*   requirements.txt

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

3.  **Activate the virtual environment:**

    ```bash
    source venv/bin/activate
    ```

4.  **Edit the variables:**

    Modify the variables at the top of the [`train_wake_word.py`](train_wake_word.py) file to configure the training process.

3.  **Run the training script:**

    ```bash
    python train_wake_word.py
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

These dependencies are installed by the [`train_wake_word.setup.sh`](train_wake_word.setup.sh) script.

*   TensorFlow (CPU or GPU)
*   torch<=2.7
*   torchvision
*   torchaudio
*   webrtcvad
*   protobuf==3.20.3
*   numpy==1.26.4
*   mutagen==1.47.0
*   torchinfo==1.8.0
*   torchmetrics==1.2.0
*   speechbrain==0.5.14
*   audiomentations==0.33.0
*   torch-audiomentations==0.11.0
*   acoustics==0.2.6
*   onnx\_tf==1.10.0
*   onnx2tf
*   onnx
*   ai\_edge\_litert==1.2.0
*   onnx\_graphsurgeon
*   sng4onnx
*   pronouncing==0.2.0
*   datasets==2.14.6
*   deep-phonemizer==0.0.19
*   piper-phonemize==1.1.0
*   tensorflow\_probability==0.16.0
*   PyYAML

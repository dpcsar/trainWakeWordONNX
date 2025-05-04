#!/usr/bin/env python3
# coding: utf-8

"""
OpenWakeWord Custom Model Training Script

This script trains a custom wake word model for OpenWakeWord.
Adapted from the Colab notebook version to run in WSL.

Setup the environment:
python3.10 -m venv venv
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
"""
# Use tensorflow cpu?
tf_cpu = True
# Define wake word parameters
target_words = ['jarvis', 'hey_jarvis']  # Variants
model_name = 'jarvis'
# Training parameters
# Each paramater controls a different aspect of training:
# `number_of_examples` controls how many examples of your wakeword
# generated. The default (1,000) usually produces a good model,
# but between 30,000 and 50,000 is often the best.
# range: min: 100, max: 50000
number_of_examples = 50000

# `number_of_training_steps` controls how long to train the model.
# Similar to the number of examples, the default (10,000) usually works well
# but training longer usually helps.
# range: min: 0, max: 50000
number_of_training_steps = 50000

# `false_activation_penalty` controls how strongly false activations
# are penalized during the training process. Higher values can make the model
# less likely to activate when it shouldn't, but may also cause it
# to not activate when the wake word isn't spoken clearly and there is
# background noise.
# range: min: 100, max: 5000
false_activation_penalty = 500
 
import os
import sys
import subprocess
import locale
from pathlib import Path

def setup_environment():
    """Set up the environment, install dependencies and clone repositories."""
    print("="*80)
    print("Setting up environment...")
    print("="*80)
    
    # Set up UTF-8 encoding
    def getpreferredencoding(do_setlocale=True):
        return "UTF-8"
    locale.getpreferredencoding = getpreferredencoding

    # Clone piper-sample-generator if needed
    if not os.path.exists("./piper-sample-generator"):
        print("Cloning piper-sample-generator...")
        subprocess.run('git clone https://github.com/rhasspy/piper-sample-generator', shell=True)
        subprocess.run("wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'", shell=True)
        #aternative subprocess.run("wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/en_US-libritts_r-medium.pt'", shell=True)
        
        # Edit generate_samples.py to set weights_only=False
        print("Modifying generate_samples.py to use weights_only=False...")
        with open("./piper-sample-generator/generate_samples.py", "r") as f:
            content = f.read()
        
        if "torch.load(model_path)" in content:
            modified = content.replace("torch.load(model_path)", "torch.load(model_path, weights_only=False)")
            with open("./piper-sample-generator/generate_samples.py", "w") as f:
                f.write(modified)
            print("Successfully modified generate_samples.py")
        else:
            print("Could not find weights_only parameter in generate_samples.py")
        
    # Install basic dependencies
    print("Installing basic dependencies...")
    dependencies = [
        # TODO: -fix? needed? 'pip install piper-phonemizer',
        'pip install "torch<=2.7"',
        'pip install "torchvision"',
        'pip install "torchaudio"',
        'pip install webrtcvad',
        'pip install "protobuf==3.20.3"',
        'pip install "numpy==1.26.4"',
    ]
    for package in dependencies:
        subprocess.run(package, shell=True)
 
    # Clone openwakeword if needed
    if not os.path.exists("./openwakeword"):
        print("Cloning openwakeword...")
        subprocess.run('git clone https://github.com/dscripka/openwakeword', shell=True)
        subprocess.run('pip install -e ./openwakeword', shell=True)
    # Clean the repository
    # subprocess.run('cd ./openwakeword && git clean -xdff', shell=True)

    # Add to path
    if "piper-sample-generator/" not in sys.path:
        sys.path.append("piper-sample-generator/")

    # Install additional dependencies
    print("Installing additional dependencies...")
    dependencies = [
        'pip install "mutagen==1.47.0"',
        'pip install "torchinfo==1.8.0"',
        'pip install "torchmetrics==1.2.0"',
        'pip install "speechbrain==0.5.14"',
        'pip install "audiomentations==0.33.0"',
        'pip install "torch-audiomentations==0.11.0"',
        'pip install "acoustics==0.2.6"',
        'pip install "onnx_tf==1.10.0"',
        'pip install "onnx2tf"',
        'pip install "onnx"',
        'pip install "ai_edge_litert==1.2.0"',
        'pip install "onnx_graphsurgeon"',
        'pip install "sng4onnx"',
        'pip install "pronouncing==0.2.0"',
        'pip install "datasets==2.14.6"',
        'pip install "deep-phonemizer==0.0.19"',
        'pip install "piper-phonemize==1.1.0"',
        'pip install "tensorflow_probability==0.16.0"',
    ]
    for package in dependencies:
        subprocess.run(package, shell=True)

    # Install TensorFlow CPU or GPU
    print("Installing TensorFlow CPU...")
    if tf_cpu:
        subprocess.run('pip install tensorflow-cpu==2.8.1', shell=True)
    else:
        subprocess.run('pip install tensorflow-gpu==2.8.1', shell=True)
 
    # Download required models
    if not os.path.exists("./openwakeword/openwakeword/resources/models"):
        print("Downloading required models...")
        os.makedirs("./openwakeword/openwakeword/resources/models")
        subprocess.run('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx', shell=True)
        subprocess.run('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite', shell=True)
        subprocess.run('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx', shell=True)
        subprocess.run('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite', shell=True)

def test_tts(target_words):
    """Generate test TTS samples for all target wake words."""
    from generate_samples import generate_samples # type: ignore
    
    # Handle both single string and list inputs
    if isinstance(target_words, str):
        target_words = [target_words]
    
    sample_files = []
    
    # Generate samples for each word in target_words
    for i, word in enumerate(target_words):
        print(f"Generating test audio for wake word [{i+1}/{len(target_words)}]: '{word}'")
        filename = f"test_generation_{word.replace(' ', '_')}.wav"
        
        generate_samples(
            text=word,
            max_samples=1,
            length_scales=[1.1],
            noise_scales=[0.7], 
            noise_scale_ws=[0.7],
            output_dir='./', 
            batch_size=1, 
            auto_reduce_batch_size=True,
            file_names=[filename]
        )
        
        sample_files.append(filename)
        print(f"Test audio saved to {filename}")
    
    # Add loop to play audio samples until user exits
    while True:
        try:
            print("\nAvailable samples:")
            for i, filename in enumerate(sample_files):
                word = filename.replace("test_generation_", "").replace(".wav", "").replace("_", " ")
                print(f"  [{i+1}] {word}")
            
            print("  [q] Quit listening")
            
            choice = input("\nSelect sample to play (number or q): ").strip().lower()
            
            if choice == 'q':
                print("Exiting audio playback loop")
                break
            
            try:
                index = int(choice) - 1
                if 0 <= index < len(sample_files):
                    print(f"Playing {sample_files[index]}...")
                    subprocess.run(f'aplay {sample_files[index]}', shell=True)
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a number or 'q' to quit")
                
        except KeyboardInterrupt:
            print("\nExiting audio playback loop")
            break

def download_data():
    """Download datasets needed for training."""
    print("Downloading training data...")
    
    # Import necessary libraries
    import datasets # type: ignore
    import scipy # type: ignore
    import numpy as np # type: ignore
    from tqdm import tqdm # type: ignore
 
    # Download MIT RIR data
    output_dir = "./mit_rirs"
    if not os.path.exists(output_dir):
        print("Downloading MIT room impulse responses...")
        os.mkdir(output_dir)
        subprocess.run('git lfs install', shell=True)
        subprocess.run('git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses', shell=True)
        
        print("Processing MIT RIR data...")
        rir_dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")]}
        ).cast_column("audio", datasets.Audio())
        
        # Save clips to 16-bit PCM wav files
        print("Converting RIR files...")
        for row in tqdm(rir_dataset):
            name = row['audio']['path'].split('/')[-1]
            scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    
    # Download AudioSet data
    if not os.path.exists("audioset"):
        print("Downloading AudioSet data...")
        os.mkdir("audioset")
        
        fname = "bal_train09.tar"
        out_dir = f"audioset/{fname}"
        link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
        subprocess.run(f'wget -O {out_dir} {link}', shell=True)
        subprocess.run('cd audioset && tar -xvf bal_train09.tar', shell=True)
        
        if not os.path.exists("./audioset_16k"):
            os.mkdir("./audioset_16k")
        
        # Save clips to 16-bit PCM wav files
        print("Converting AudioSet files...")
        audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]})
        audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
        for row in tqdm(audioset_dataset):
            name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
            scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    
    # Download Free Music Archive data
    if not os.path.exists("./fma"):
        os.mkdir("./fma")
        
        print("Downloading FMA music dataset...")
        fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
        fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))
        
        # Save clips to 16-bit PCM wav files
        n_hours = 1  # use only 1 hour of clips
        print(f"Converting {n_hours} hour(s) of music...")
        for i in tqdm(range(n_hours*3600//30)):  # FMA dataset is all 30 second clips
            row = next(fma_dataset)
            name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
            scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    
    # Download pre-computed openWakeWord features
    if not os.path.exists("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
        print("Downloading pre-computed training features...")
        subprocess.run('wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy', shell=True)
    
    # Download validation set
    if not os.path.exists("validation_set_features.npy"):
        print("Downloading validation data...")
        subprocess.run('wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy', shell=True)

def train_model(target_words_list, model_name, number_of_examples, number_of_training_steps, false_activation_penalty):
    """Train the wake word model with the specified parameters."""
    print("Preparing model training configuration...")
    
    # Import yaml here after setup environment has installed it
    import yaml # type: ignore
    
    # Load default YAML config file for training
    config = yaml.load(open("openwakeword/examples/custom_model.yml", 'r').read(), yaml.Loader)
    
    # Modify values in the config
    config["target_phrase"] = target_words_list
    config["model_name"] = model_name.replace(" ", "_")
    config["n_samples"] = number_of_examples
    config["n_samples_val"] = max(500, number_of_examples//10)
    config["steps"] = number_of_training_steps
    config["target_accuracy"] = 0.5
    config["target_recall"] = 0.25
    config["output_dir"] = "./my_custom_model"
    config["max_negative_weight"] = false_activation_penalty
    config["background_paths"] = ['./audioset_16k', './fma']
    config["false_positive_validation_data_path"] = "validation_set_features.npy"
    config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

    # Save the modified config to a new YAML file 
    with open('my_model.yaml', 'w') as file:
        yaml.dump(config, file)
    
    # Generate clips
    print("\n[Step 1/3] Generating training clips...")
    subprocess.run(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips', shell=True)
    
    # Augment the generated clips
    print("\n[Step 2/3] Augmenting clips...")
    subprocess.run(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips', shell=True)
    
    # Train model
    print("\n[Step 3/3] Training model...")
    subprocess.run(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model', shell=True)
    
    print(f"\nTraining complete! Model files are available at:")
    print(f"  my_custom_model/{config['model_name']}.onnx")

def main():
    """Main function to run the script."""
    print("="*80)
    print("OpenWakeWord Custom Model Training Script")
    print("="*80)
    
    print(f"\nTraining configuration:")
    print(f"  Test Target Words: {target_words}")
    print(f"  Number of examples: {number_of_examples}")
    print(f"  Training steps: {number_of_training_steps}")
    print(f"  False activation penalty: {false_activation_penalty}")
    
    # Setup environment and install dependencies
    setup_environment()
    
    # Generate test TTS sample
    test_tts(target_words)
    
    # Ask if user wants to continue or exit
    while True:
        continue_choice = input("\nDo you want to continue with model training? (y/n): ").strip().lower()
        if continue_choice == 'y':
            print("\nContinuing with model training...")
            break
        elif continue_choice == 'n':
            print("\nExiting training script.")
            return
        else:
            print("Please enter 'y' to continue or 'n' to exit.")
    
    # Download data
    download_data()
   
    # Start training
    train_model(target_words, model_name, number_of_examples, number_of_training_steps, false_activation_penalty)

if __name__ == "__main__":
    main()

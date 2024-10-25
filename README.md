
# Talk to Text: Speech Recognition Model

This repository contains the code, dataset, and fine-tuned model for a speech recognition project using [Whisper](https://huggingface.co/models?other=whisper) from Hugging Face. The project aims to convert speech (audio files) to text by fine-tuning a pre-trained Whisper model.

## Repository Structure

```
📁 Project data/speech_recognition_dataset
   ├── Audio files (.wav or .mp3)
   └── Corresponding text files (.txt)

📁 deployment
   ├── app.py
   ├── img/
   │   └── img2.png
   ├── model.py
   └── whisper_finetuned_V2/
       ├── Model files after fine-tuning Whisper model
       └── Checkpoints, configuration files, and tokenizer
           
📄 Triple_T_Fine_Tuned_Whisper-small.ipynb
   └── Jupyter notebook containing preprocessing, training, model saving, and evaluation code

📄 T3_Production_V2_FT_Whisper-small.ipynb
   └── Jupyter notebook containing the final pipe-line usen in inference 

📄 requirements.txt
📄 README.md
📄 team presentation/Triple T.pptx
   └── a power-point presentation showing our team members' names
```

### Contents

- **`Project data/speech_recognition_dataset/`**: Contains the dataset used for training, including audio files and their corresponding transcriptions.
- **`deployment/app.py`**: Application script for deploying the fine-tuned model.
- **`deployment/model.py`**: Model file containing the setup and loading of the Whisper model for inference.
- **`deployment/whisper_finetuned_V2/`**: Folder storing the fine-tuned Whisper model, checkpoints, configuration files, and tokenizer settings.
- **`requirements.txt`**: Lists necessary packages for the project.

## Model

The model was fine-tuned from the [OpenAI Whisper-small](https://huggingface.co/openai/whisper-small) checkpoint on a custom dataset of speech and corresponding transcriptions.

### Key Features:
- **Fine-tuned model**: Trained specifically on the dataset available in the `speech_recognition_dataset` folder.
- **Notebook**: Contains the full pipeline including preprocessing, training, and evaluation.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```

2. **Set Up the Environment**:
   Install the required packages (make sure to include Hugging Face's Transformers, librosa, and evaluate):

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can manually install the key dependencies:

   ```bash
   pip install transformers librosa evaluate
   ```

3. **Run the Jupyter Notebook**:
   Open the `Talk To Text.ipynb` notebook and run each cell to preprocess the data, fine-tune the model, and evaluate the results.

   ```bash
   jupyter notebook Talk To Text.ipynb
   ```

## Dataset

The dataset consists of audio files in `.wav` or `.mp3` format and their corresponding text transcriptions in `.txt` files. These are located in the `speech_recognition_dataset` directory.

### Preprocessing:
The audio data is preprocessed using librosa to convert the audio into the format required by the Whisper model. The notebook demonstrates how to load, preprocess, and organize this data for model training.

## Evaluation

The evaluation of the model is done using the **Word Error Rate (WER)** metric, which is calculated by comparing the model’s predicted transcription to the ground truth transcription.

## Team Members
- [Essam Omar](https://github.com/eoabdulhalim)
- [Mostafa Ashraf](https://github.com/M0STAFA-A4F)
- [Mahmoud Amin](https://github.com/mahmoudamintaha)
- Salem El-Sayed
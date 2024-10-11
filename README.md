
# Talk to Text: Speech Recognition Model

This repository contains the code, dataset, and fine-tuned model for a speech recognition project using [Whisper](https://huggingface.co/models?other=whisper) from Hugging Face. The project aims to convert speech (audio files) to text by fine-tuning a pre-trained Whisper model.

## Repository Structure

```
📁 Project data/speech_recognition_dataset
   ├── Audio files (.wav or .mp3)
   └── Corresponding text files (.txt)

📁 fine-tuned model
   └── Model files after fine-tuning Whisper model

📄 Talk To Text.ipynb
   └── Jupyter notebook containing preprocessing, training, model saving, and evaluation code
```

### Contents
- **`Project data/speech_recognition_dataset/`**: This folder contains the dataset used for training, which includes audio files and their corresponding transcriptions.
- **`fine-tuned model/`**: This folder stores the Whisper model after fine-tuning on the dataset.
- **`Talk To Text.ipynb`**: A comprehensive notebook that walks through the steps of:
  1. Preprocessing the dataset (loading audio files, extracting features, and preparing text labels).
  2. Fine-tuning the Whisper model.
  3. Saving the trained model for inference.
  4. Evaluating the model's performance on test data.

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

## Contributions

Feel free to open issues or pull requests if you'd like to contribute!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

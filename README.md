# Predicting and Generating Video Sequences Using Deep Learning

## Overview
This project focuses on predicting and generating video sequences using deep learning models. The primary goal is to predict future video frames based on short input sequences from the UCF101 dataset, effectively simulating ongoing actions in the scene. The models implemented include CNN-LSTM and PredRNN, which excel at spatial-temporal and advanced temporal modeling, respectively.

## Features
- **Data Preparation**: Preprocessed UCF101 video frames into manageable formats (64x64, grayscale/RGB).
- **Model Implementation**:
  - CNN-LSTM for capturing spatial-temporal dependencies.
  - PredRNN for advanced temporal modeling.
- **Video Generation**: Generated continuous video sequences from predicted frames using OpenCV.
- **User Interface**: Interactive UI (Streamlit) to visualize predictions, input frames, and complete video sequences.
- **Evaluation Metrics**: Frame prediction quality assessed using MSE and SSIM.

## Dataset
The project uses the UCF101 dataset, a collection of videos categorized by human activities.
- [UCF101 Dataset on Kaggle](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition/data)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-prediction-dl.git
   ```
2. Navigate to the project directory:
   ```bash
   cd video-prediction-dl
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preparation**:
   - Run the preprocessing script to prepare the UCF101 dataset.
   ```bash
   python preprocess_data.py
   ```
2. **Model Training**:
   - Train CNN-LSTM or PredRNN models by running the respective scripts.
   ```bash
   python train_cnn_lstm.py
   python train_predrnn.py
   ```
3. **Video Generation**:
   - Use the video generation script to create videos from predicted frames.
   ```bash
   python generate_video.py
   ```
4. **User Interface**:
   - Launch the interactive UI to visualize input and predicted frames.
   ```bash
   streamlit run app.py
   ```

## Results
- **CNN-LSTM**: Efficient at capturing spatial-temporal patterns, suitable for basic motion prediction.
- **PredRNN**: Outperformed CNN-LSTM in long-term motion prediction with superior temporal modeling.

## Tools and Technologies
- **Programming Languages**: Python
- **Libraries**: TensorFlow, PyTorch, OpenCV, Streamlit
- **Dataset**: UCF101

## Contributions
Feel free to contribute by submitting issues or pull requests. Make sure to follow the contribution guidelines.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For queries or suggestions, feel free to reach out:
- **Email**: m.abdullahh.1385@gmail.com
- **GitHub**: [Afkayyy](https://github.com/Afkayyy)

---

### Acknowledgments
Special thanks to the creators of the UCF101 dataset and the deep learning frameworks used in this project.

# 🎬 Video Subtitle Generator

A **PyQt5-based** desktop application that automatically generates subtitles for video files using **OpenAI's Whisper model**. The application supports batch processing of videos and multiple languages.

## ✨ Features

- 📂 **Process single video files or entire directories**
- 🎥 **Support for multiple video formats** (MP4, AVI, MOV, MKV, FLV, WMV)
- ⚡ **Batch processing with parallel execution**
- 🗣 **Multiple Whisper model options** (tiny, base, small, medium, large)
- 🌍 **Multi-language support**
- 🌙 **Dark mode interface**
- 📊 **Real-time progress tracking**
- 📜 **Detailed processing logs**
- 🛠 **Error handling and retry mechanism**
- ⚙ **Float16/32 precision options**
- ❌ **Cancel processing capability**

## 📋 Requirements

- **Python 3.8+**
- **PyQt5**
- **OpenAI Whisper**
- **FFmpeg**
- **PyTorch**
- Other dependencies (see `requirements.txt`)

## 🔧 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/cybobug/Subtitle-Generator
   ```

2. **Install FFmpeg**:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Run the Application**:
   ```bash
   python subtitle_generator.py
   ```

2. **Select Input**:
   - Choose either a **single video file** or a **directory** containing multiple videos.
   - Select the desired **Whisper model** (larger models are more accurate but slower).
   - Choose the **target language** for transcription.
   - *(Optional)* Enable **low precision mode** for faster processing.

3. **Click "Generate Subtitles"** to start processing.

4. **Monitor Progress**:
   - **Real-time progress updates** in the UI.
   - **Detailed logs** in the text display.
   - **Cancel processing** at any time if needed.

5. **Results**:
   - **SRT files** are generated in the `subtitle_output` directory.
   - Summary of **successful/failed processes** is displayed.
   - Option to **open output directory automatically**.

## 🤝 Contributing

1. **Fork the Repository**.
2. **Create a New Branch**:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit Your Changes**:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to Your Branch**:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**.

## 🛡 License

This project is licensed under the **MIT License** – see the LICENSE file for details.

## 🙌 Acknowledgments

- 🎤 [OpenAI Whisper](https://github.com/openai/whisper) – Speech recognition model.
- 🖥 [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) – GUI framework.
- 🎵 [FFmpeg](https://ffmpeg.org/) – Audio processing.

---

🎬 **Enhance your video content with accurate subtitles!** 🚀


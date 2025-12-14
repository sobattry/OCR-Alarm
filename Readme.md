# Screen OCR Monitor

A lightweight, modern desktop utility for Windows that periodically captures a specific region of your screen, runs deep-learning-based OCR (Optical Character Recognition), and triggers automated actions when specific text is detected.

Built with **Python**, **PyQt6**, and **RapidOCR**.

## Features

* **Real-time Monitoring:** Captures a screen region at a user-defined frequency (e.g., every 2 seconds).
* **Deep Learning OCR:** Uses `RapidOCR` (ONNX) for high-accuracy text recognition, far superior to standard Tesseract for screen text.
* **Smart Filtering:**
    * **Keyword Detection:** Trigger actions only when specific text is found.
    * **Fuzzy Matching:** Detects keywords even if there are small OCR errors (e.g., "Login" vs "Log1n").
    * **Change Detection:** Pauses OCR if the screen hasn't changed to save CPU.
* **Automated Actions:**
    * **Telegram:** Send the detected text and/or a screenshot to your Telegram chat.
    * **Email:** Send an email alert with the text and screenshot attachment.
    * **Local Logging:** Save detection history to a JSON log file.
* **Modern GUI:**
    * Dark-themed, minimal interface.
    * Visual "Select Region" overlay tool.
    * Configurable options (stop on match, include/exclude screenshots).

## Installation

### Prerequisites
* Python 3.8 or higher.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/screen-ocr-monitor.git](https://github.com/yourusername/screen-ocr-monitor.git)
cd screen-ocr-monitor
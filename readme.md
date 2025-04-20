
# 🦩 Chatlama - Ollama Model Chat Interface

**Chatlama** is a user-friendly Streamlit app that allows you to interact with locally running [Ollama](https://ollama.com/) language models. You can upload Word documents, ask questions based on the content, and monitor model performance and system metrics—all in one elegant UI.

---

## 🚀 Features

- 🔍 **Select any local Ollama model** via dropdown  
- 💬 **Chat** with the selected model directly  
- 📄 **Upload** `.docx` **documents** and ask questions based on their content  
- ⏱️ Displays **response time** and **total time**  
- 🖥️ Shows **system info** (CPU, memory, OS, usage) in the sidebar  

---

## ⚙️ Prerequisites

Make sure you have the following before running the app:

1. Install [Ollama](https://ollama.com/download)
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull a model (e.g., LLaMA 3):
   ```bash
   ollama pull llama3
   ```

---

## 💪 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/chatlama.git
cd chatlama
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

Make sure Ollama is running locally on port `11434` with your models loaded.

```bash
streamlit run app.py
```

---

## 📾 Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally with at least one model loaded (e.g., `llama3`)
- Word documents (`.docx`) for document-based Q&A




---

## 📁 Project Structure

```
chatlama/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation
```

---

## 🙌 Acknowledgements

- [Ollama](https://ollama.com/)

---

## 📜 License

MIT License – feel free to use, modify, and distribute.

---

## 🔗 Connect

Built with ❤️ to explore local LLMs. For questions or collaboration, feel free to reach out!


# 📜 Legal Chatbot

A **legal document-based chatbot** that can extract, process, and summarize legal PDFs using **LangChain, Ollama, and ChromaDB**. This bot provides instant responses to legal queries based on uploaded documents.

## 🌍 Live Deployment
Deployed on Railway:
🔗 [Legal Chatbot](https://legal-chatbot-production.up.railway.app/)

## ✨ Features
- 📂 **Upload Legal Documents** (PDF, TXT) for analysis
- 🔎 **Retrieve & Answer Questions** from the documents
- 🧠 **Summarize Complex Legal Text** into easy-to-understand language
- 💾 **Store & Index Data** using ChromaDB for efficient retrieval
- 🤖 **Multi-LLM Support** (Llama3, DeepSeek, Phi4 via Ollama API)
- 🖥 **Interactive Chat Interface** with a clean UI

---

## 🚀 Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **LLM:** Ollama (Llama3, DeepSeek, Phi4)
- **Embeddings:** LangChain OllamaEmbeddings
- **Vector Store:** ChromaDB
- **PDF Processing:** Unstructured, pdfplumber, PyMuPDF
- **Deployment:** Railway.app

---

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/SamarthShinde/legal-chatbot.git
cd legal-chatbot
```

### 2️⃣ Create & Activate Conda Environment
```sh
conda create --name legal-chatbot python=3.10 -y
conda activate legal-chatbot
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```sh
streamlit run app.py
```

---

## 🚢 Deployment on Railway
### 1️⃣ Push to GitHub
```sh
git add .
git commit -m "Deploy Legal Chatbot"
git push origin main
```

### 2️⃣ Deploy on Railway
1. Go to [Railway.app](https://railway.app/)
2. Create a **New Project** → Deploy from GitHub
3. Connect your repository
4. **Set Up Environment Variables**
   ```sh
   OLLAMA_BASE_URL=https://ollama.thecit.in
   ```
5. Use the following **Start Command** in `Procfile` (if using Nixpacks):
   ```sh
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
6. Click **Deploy** 🚀

---

## 📜 Example Usage
1. Upload a **legal PDF**
2. Ask questions like:
   - *"What are the key legal provisions mentioned?"*
   - *"Summarize the plaintiff’s arguments."*
3. Get a structured response instantly ✅

---

## 🛠 Future Enhancements
- 🔍 **Improve OCR Accuracy** for scanned documents
- 🔗 **Integrate OpenAI GPT for Legal Reasoning**
- 📊 **Enhance UI with Custom Chat Components**

---

## 🤝 Contributing
Want to contribute? Open a PR or raise an issue!

📧 **Contact:** Samarth Shinde - [GitHub](https://github.com/SamarthShinde)

---

🚀 **Enjoy Instant Legal Insights!**


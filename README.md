# 🔍 Deep Research AI Agentic System

An efficient **multi-agent research assistant** built using **LangChain**, **LangGraph**, **Tavily**, and **Groq's Llama 3 models**.

🚀 Enter a research question → Collect real-time data → Draft a high-quality answer automatically.

---

## 🛠 Features

- **Multi-Agent Architecture**  
  📚 Research Agent (Tavily) for web data collection  
  ✍️ Drafting Agent (Groq Llama3) for writing summarized answers
- **Beautiful Streamlit UI** with smooth user experience
- **Download Drafted Answers** as text files
- **View Past Research Drafts** from session history
- **Error Handling and Status Indicators** for robustness

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/deep-research-ai.git
   cd deep-research-ai

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
3. **Set Your API Keys**: 
    Make sure you have:
    GROQ_API_KEY (for Llama3 models)
    TAVILY_API_KEY (for web search)
4. **Run the Application**
     ```bash
   streamlit run ui.py



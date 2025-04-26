import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deep Research AI", layout="wide")
st.title("🔍 Deep Research AI Agentic System")
st.markdown("""
Welcome to **Deep Research AI** — an efficient, multi-agent research assistant built using **LangGraph**, **LangChain**, **Tavily**, and **Groq Llama3**.

🚀 Enter a research question, and let our agents work their magic!
""")

# --- USER INPUT ---
with st.sidebar:
    st.header("Configuration ⚙️")
    query = st.text_input("🧠 Enter your research question:")
    num_results = st.slider("🔎 Number of search results:", min_value=3, max_value=10, value=5)
    temperature = st.slider("🔥 Draft creativity (temperature):", min_value=0.0, max_value=1.0, value=0.3)
    run_research = st.button("🚀 Run Research Agents")

# --- AGENTS ---

# Research Agent (Tavily)
def tavily_research_agent(query, k=5):
    try:
        tool = TavilySearchResults(k=k)
        docs = tool.invoke({"query": query})
        return docs
    except Exception as e:
        st.error(f"Research Agent Error: {e}")
        return []

# Drafting Agent (Groq LLM)
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
)

prompt_template = ChatPromptTemplate.from_template("""
You are a highly skilled research assistant AI.

Based on the research snippets below, write a clear, factual, and well-organized answer to the user's question.

Question:
{query}

Research Snippets:
{research}

Answer:
""")

def answer_drafter_agent(query, research_snippets):
    try:
        research_text = "\n\n".join([
            snippet.get('content') or snippet.get('snippet') or str(snippet)
            for snippet in research_snippets
        ])
        formatted_prompt = prompt_template.format_messages(query=query, research=research_text)
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        st.error(f"Drafting Agent Error: {e}")
        return None

# --- MAIN FLOW ---
if run_research and query:
    with st.spinner("🔎 Research Agent is gathering data..."):
        snippets = tavily_research_agent(query, k=num_results)

    if snippets:
        st.subheader("📚 Collected Research Snippets")
        for idx, snippet in enumerate(snippets):
            content = snippet.get('content') or snippet.get('snippet') or str(snippet)
            with st.expander(f"Snippet {idx + 1}"):
                st.write(content)

        with st.spinner("✍️ Drafting Agent is writing the final answer..."):
            final_answer = answer_drafter_agent(query, snippets)

        if final_answer:
            st.subheader("📝 Final Drafted Answer")
            st.success(final_answer)

            # Option to download
            st.download_button(
                label="💾 Download Draft as Text",
                data=final_answer,
                file_name="drafted_answer.txt",
                mime="text/plain",
            )

            # Save history
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({"query": query, "answer": final_answer})

    else:
        st.error("No research snippets were retrieved. Try again!")

# --- HISTORY SECTION ---
if "history" in st.session_state and st.session_state["history"]:
    st.sidebar.markdown("---")
    st.sidebar.header("📜 Previous Drafts")
    for idx, item in enumerate(st.session_state["history"][-3:][::-1]):
        with st.sidebar.expander(f"Draft #{len(st.session_state['history']) - idx}"):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Answer:** {item['answer']}")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "Built with ❤️ by Keyur Doshi using **LangGraph**, **LangChain**, **Tavily**, and **Groq**."
)
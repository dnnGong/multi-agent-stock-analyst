from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from openai import OpenAI

# ==============================
# Environment / config
# ==============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "stocks.db"
DB_PATH = os.getenv("STOCK_AGENTS_DB_PATH", str(DEFAULT_DB_PATH))

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Please export it before running Streamlit.")
    st.stop()
if not ALPHAVANTAGE_API_KEY:
    st.error("Missing ALPHAVANTAGE_API_KEY. Please export it before running Streamlit.")
    st.stop()
if not Path(DB_PATH).exists():
    st.error(
        "Missing stocks.db. Put your sqlite file at app_test/data/stocks.db "
        "or set STOCK_AGENTS_DB_PATH to your db path."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# ==============================
# Tools
# ==============================
def get_price_performance(tickers: list[str], period: str = "1y") -> dict:
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price": round(end, 2),
                "pct_change": round((end - start) / start * 100, 2),
                "period": period,
            }
        except Exception as exc:
            results[ticker] = {"error": str(exc)}
    return results


def get_market_status() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=15,
    ).json()


def get_top_gainers_losers() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=15,
    ).json()


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=15,
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title": a.get("title"),
                "source": a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score": a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }


def query_local_db(sql: str) -> dict:
    if not sql.strip().lower().startswith("select"):
        return {"error": "Only SQL SELECT statements are allowed."}
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as exc:
        return {"error": str(exc)}


# def get_company_overview(ticker: str) -> dict:
#     # Company fundamentals tool
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info

#         pe_ratio = info.get("forwardPE") or info.get("trailingPE")
#         market_cap = info.get("marketCap")
#         company_name = info.get("longName") or info.get("shortName")

#         if pe_ratio is None and market_cap is None:
#             return {"error": f"No fundamental data found for {ticker}"}

#         return {
#             "ticker": ticker,
#             "name": company_name if company_name else "N/A",
#             "pe_ratio": round(pe_ratio, 2) if pe_ratio else "N/A",
#             "market_cap": market_cap if market_cap else "N/A",
#         }
#     except Exception as exc:
#         return {"error": f"Error fetching data for {ticker}: {str(exc)}"}

def get_company_overview(ticker: str) -> dict:
    # Company fundamentals tool using Alpha Vantage instead of yfinance
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
        response = requests.get(url, timeout=15)
        data = response.json()

        # 处理 Alpha Vantage 的 API 调用频率限制 (Rate Limit) 提示
        if "Information" in data and "rate limit" in data["Information"].lower():
            return {"error": "Alpha Vantage API rate limit reached. Please try again later."}

        # 如果返回的数据没有 Symbol，说明 Ticker 无效或没查到数据
        if not data or "Symbol" not in data:
            return {"error": f"No fundamental data found for {ticker}. The ticker might be invalid."}

        pe_ratio = data.get("PERatio")
        market_cap = data.get("MarketCapitalization")
        company_name = data.get("Name")

        # 安全地格式化 P/E Ratio
        pe_formatted = "N/A"
        if pe_ratio and pe_ratio != "None":
            try:
                pe_formatted = round(float(pe_ratio), 2)
            except ValueError:
                pe_formatted = pe_ratio

        return {
            "ticker": data.get("Symbol", ticker),
            "name": company_name if company_name else "N/A",
            "pe_ratio": pe_formatted,
            "market_cap": market_cap if market_cap else "N/A",
        }
    except Exception as exc:
        return {"error": f"Error fetching data for {ticker}: {str(exc)}"}


def get_tickers_by_sector(sector: str) -> dict:
    # Sector/industry ticker lookup tool
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT ticker FROM stocks WHERE sector = ?"
        df = pd.read_sql_query(query, conn, params=(sector,))

        if df.empty:
            query = "SELECT ticker FROM stocks WHERE industry LIKE ?"
            df = pd.read_sql_query(query, conn, params=(f"%{sector}%",))

        if df.empty:
            return {"stocks": []}

        return {"stocks": df["ticker"].tolist()}
    finally:
        conn.close()


ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector": get_tickers_by_sector,
    "get_price_performance": get_price_performance,
    "get_company_overview": get_company_overview,
    "get_market_status": get_market_status,
    "get_top_gainers_losers": get_top_gainers_losers,
    "get_news_sentiment": get_news_sentiment,
    "query_local_db": query_local_db,
}


# ==============================
# Tool schemas
# ==============================
def _s(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


SCHEMA_TICKERS = _s("get_tickers_by_sector", "Find stocks by sector/industry.", {"sector": {"type": "string"}}, ["sector"])
SCHEMA_PRICE = _s(
    "get_price_performance",
    "Get percentage price performance for tickers over a period.",
    {"tickers": {"type": "array", "items": {"type": "string"}}, "period": {"type": "string", "default": "1y"}},
    ["tickers"],
)
SCHEMA_OVERVIEW = _s("get_company_overview", "Get company fundamentals for one ticker.", {"ticker": {"type": "string"}}, ["ticker"])
SCHEMA_STATUS = _s("get_market_status", "Get market open/close status.", {}, [])
SCHEMA_MOVERS = _s("get_top_gainers_losers", "Get top gainers/losers and most active.", {}, [])
SCHEMA_NEWS = _s("get_news_sentiment", "Get news sentiment for a ticker.", {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, ["ticker"])
SCHEMA_SQL = _s("query_local_db", "Run SQL SELECT on local stocks DB.", {"sql": {"type": "string"}}, ["sql"])

ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW, SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]


# ==============================
# Agent core
# ==============================
@dataclass
class AgentResult:
    agent_name: str
    answer: str
    tools_called: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    issues_found: list[str] = field(default_factory=list)
    reasoning: str = ""


def run_specialist_agent(
    model: str,
    agent_name: str,
    system_prompt: str,
    task: str,
    tool_schemas: list,
    max_iters: int = 8,
    verbose: bool = False,
) -> AgentResult:
    start_time = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tools_called: list[str] = []
    raw_data: dict[str, Any] = {}
    final_answer = ""

    iterations = 0
    while iterations < max_iters:
        iterations += 1

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            temperature=0,
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in response_message.tool_calls[:25]
                    ],
                }
            )

            for tool_call in response_message.tool_calls[:25]:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    func_args = {}

                if verbose:
                    print(f"[{agent_name}] Calling: {func_name}({func_args})")

                if func_name in ALL_TOOL_FUNCTIONS:
                    try:
                        function_response = ALL_TOOL_FUNCTIONS[func_name](**func_args)
                    except Exception as exc:
                        function_response = {"error": str(exc)}
                else:
                    function_response = {"error": f"Tool {func_name} not found."}

                tools_called.append(func_name)
                raw_data[f"{func_name}:{len(tools_called)}"] = function_response

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps(function_response),
                    }
                )
            continue

        final_answer = response_message.content or ""
        break

    duration = time.time() - start_time

    return AgentResult(
        agent_name=agent_name,
        answer=final_answer if final_answer else "Agent failed to provide an answer.",
        tools_called=tools_called,
        raw_data=raw_data,
        reasoning=f"Completed in {iterations} iterations. Time taken: {duration:.2f}s",
    )


def run_single_agent(question: str, model: str, verbose: bool = False) -> AgentResult:
    system_prompt = """You are a professional financial analyst with access to real-time tools.

CRITICAL INSTRUCTIONS:
1. DATA FIRST: Always use provided tools to fetch real-time data.
2. MULTI-STEP LOGIC: For multi-condition questions, verify each condition explicitly.
3. MATHEMATICAL SORTING: If asked for Top/Best/Worst, compare numerical values correctly.
4. NO HALLUCINATIONS: If a tool returns error/no data, state it clearly.
5. EVIDENCE-BASED ANSWERS: Include specific numbers supporting the conclusion.
"""

    return run_specialist_agent(
        model=model,
        agent_name="Single Agent",
        system_prompt=system_prompt,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        max_iters=10,
        verbose=verbose,
    )


def run_multi_agent(question: str, model: str, verbose: bool = False) -> dict:
    start_time = time.time()
    agents_activated_results: list[AgentResult] = []

    # Orchestrator (matches your notebook logic)
#     orch_prompt = f"""
# Analyze the user question: "{question}"
# Which specialists are needed? Respond ONLY with a comma-separated list of categories:
# 'Fundamentals', 'MarketData', 'Sentiment'.
# """
    # orch_prompt = f"""
    # Analyze the user question: "{question}"
    # Determine which financial specialists are required.
    # - 'Fundamentals': For P/E, Market Cap, Sector lookup, or company info.
    # - 'MarketData': For stock prices, % change, performance, or market status.
    # - 'Sentiment': For news and market mood.

    # Respond with a comma-separated list. If unsure, include both 'Fundamentals' and 'MarketData'.
    # """
    orch_prompt = f"""
    Analyze the user question: "{question}"
    Determine which financial specialists are required.
    - 'Fundamentals': For P/E, Market Cap, Sector lookup, or company info.
    - 'MarketData': For stock prices, % change, performance, or market status.
    - 'Sentiment': For news and market mood.

    ---
    EXAMPLE:
    User Question: "What is the P/E ratio and current price of Apple?"
    Response: Fundamentals, MarketData
    ---

    Respond ONLY with a comma-separated list of the required specialists. 
    If unsure, include both 'Fundamentals' and 'MarketData'.
    """
    orch_res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial task orchestrator."},
            {"role": "user", "content": orch_prompt},
        ],
        temperature=0,
    )
    decision = orch_res.choices[0].message.content or ""

    # if "Fundamentals" in decision:
    #     fund_tools = [s for s in ALL_SCHEMAS if s["function"]["name"] in ["get_company_overview", "get_tickers_by_sector"]]
    #     res = run_specialist_agent(
    #         model,
    #         "Fundamentals Specialist",
    #         "You are an expert in company fundamentals. Use tools to get P/E, Market Cap, etc.",
    #         question,
    #         fund_tools,
    #         verbose=verbose,
    #     )
    #     agents_activated_results.append(res)

    # if "Fundamentals" in decision:
    #     # 修复：增加 query_local_db 权限，使其能处理“哪些、其中、对比”类问题
    #     fund_tools = [
    #         s for s in ALL_SCHEMAS 
    #         if s["function"]["name"] in ["get_company_overview", "get_tickers_by_sector", "query_local_db"]
    #     ]
    #     res = run_specialist_agent(
    #         model,
    #         "Fundamentals Specialist",
    #         # 强化指令，强制其使用工具并允许其进行 SQL 查询
    #         "You are an expert in company fundamentals. Use get_company_overview for single stocks or query_local_db for comparing multiple stocks via SQL. NEVER say you lack data access.",
    #         question,
    #         fund_tools,
    #         verbose=verbose,
    #     )
    #     agents_activated_results.append(res)
    if "Fundamentals" in decision:
        fund_tools = [
            s for s in ALL_SCHEMAS 
            if s["function"]["name"] in ["get_company_overview", "get_tickers_by_sector", "query_local_db"]
        ]
        res = run_specialist_agent(
            model,
            "Fundamentals Specialist",
            # 强化指令：强调 Ticker 的转换
            "You are an expert in company fundamentals. CRITICAL: When using get_company_overview, you MUST use the exact stock ticker (e.g., 'AAPL', 'INTC'), NOT the company name. Use query_local_db if you need to compare multiple stocks. NEVER say you lack data access.",
            question,
            fund_tools,
            verbose=verbose,
        )
        agents_activated_results.append(res)

    if "MarketData" in decision or "Market Data" in decision:
        mkt_tools = [
            s
            for s in ALL_SCHEMAS
            if s["function"]["name"] in ["get_price_performance", "get_market_status", "get_top_gainers_losers"]
        ]
        mkt_prompt = """You are a Market Data expert.
1. Call get_price_performance ONCE for all tickers combined.
2. Once you have data, output findings immediately.
3. DO NOT repeat calls for the same ticker.
"""
        res = run_specialist_agent(
            model,
            "Market Data Specialist",
            mkt_prompt,
            question,
            mkt_tools,
            verbose=verbose,
        )
        agents_activated_results.append(res)

    if "Sentiment" in decision:
        sent_tools = [s for s in ALL_SCHEMAS if s["function"]["name"] == "get_news_sentiment"]
        res = run_specialist_agent(
            model,
            "Sentiment Specialist",
            "You are an expert in financial news sentiment. Analyze market mood.",
            question,
            sent_tools,
            verbose=verbose,
        )
        agents_activated_results.append(res)

    if not agents_activated_results:
        all_context = "No specific tool data was retrieved by specialists."
    else:
        all_context = "\n".join([f"{r.agent_name}: {r.answer}" for r in agents_activated_results])

    synth_res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a senior analyst. Synthesize specialist data into a final professional response."},
            {"role": "user", "content": f"Question: {question}\nData retrieved:\n{all_context}"},
        ],
        temperature=0,
    )

    return {
        "final_answer": synth_res.choices[0].message.content or "",
        "agent_results": agents_activated_results,
        "elapsed_sec": time.time() - start_time,
        "architecture": "Orchestrator-Specialists",
    }


# ==============================
# Streamlit app
# ==============================
st.set_page_config(page_title="MP3 Deployment App", page_icon="💬", layout="wide")
st.title("💬 MP3 Agent Chat")
st.caption("Streamlit interface")

st.sidebar.header("Controls")
agent_selector = st.sidebar.selectbox("Agent selector", ["Single Agent", "Multi-Agent"], index=1)
model_selector = st.sidebar.selectbox("Model selector", ["gpt-4o-mini", "gpt-4o"], index=0)
if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.caption(f"DB path: {DB_PATH}")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _history_text(messages: list[dict[str, Any]], max_turns: int = 3) -> str:
    # requirement: handle follow-ups up to 3 exchanges deep
    recent = messages[-(max_turns * 2):]
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])


def _rewrite_followup(model: str, history_text: str, latest_user: str) -> str:
    if not history_text.strip():
        return latest_user

    prompt = (
        "Rewrite the latest user message into a standalone finance question. "
        "Resolve references like 'that', 'it', 'the two', 'those'. "
        "Return only the rewritten question."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"History:\n{history_text}\n\nLatest user message:\n{latest_user}"},
            ],
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or latest_user
    except Exception:
        return latest_user


# render full history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(f"architecture={msg['meta']['architecture']} | model={msg['meta']['model']}")


user_prompt = st.chat_input("Ask a stock question...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = _history_text(st.session_state.messages, max_turns=3)
            standalone = _rewrite_followup(model_selector, history, user_prompt)

            # Core requirement: pass conversation history to agent every turn
            task = (
                "Use conversation context to answer the latest question accurately.\n\n"
                f"Conversation history:\n{history}\n\n"
                f"Latest user message:\n{user_prompt}\n\n"
                f"Resolved standalone question:\n{standalone}"
            )

            try:
                if agent_selector == "Single Agent":
                    out = run_single_agent(task, model_selector, verbose=False)
                    answer = out.answer
                    meta = {"architecture": "Single Agent", "model": model_selector}
                else:
                    out = run_multi_agent(task, model_selector, verbose=False)
                    answer = out.get("final_answer", "")
                    meta = {"architecture": out.get("architecture", "Multi-Agent"), "model": model_selector}
            except Exception as exc:
                answer = f"Error: {exc}"
                meta = {"architecture": agent_selector, "model": model_selector}

        st.markdown(answer)
        st.caption(f"architecture={meta['architecture']} | model={meta['model']}")
        with st.expander("Resolved follow-up question", expanded=False):
            st.write(standalone)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})

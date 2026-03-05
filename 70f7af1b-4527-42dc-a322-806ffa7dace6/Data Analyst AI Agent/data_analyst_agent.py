# ============================================================
# 🤖 Data Analyst AI Agent using OpenAI + pandas + matplotlib
# ============================================================
# This agent can:
#   - Answer questions about a dataset
#   - Write and execute Python code for analysis
#   - Generate charts and summaries
#   - Iteratively refine answers via a ReAct-style loop
# ============================================================

import os
import json
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from openai import OpenAI

# ----------------------------------------------------------
# 1️⃣  CONFIG  — replace with your key or use an env var
# ----------------------------------------------------------
OPENAI_API_KEY = ""          # <-- paste your key here
MODEL          = "gpt-4o"          # best for code generation
MAX_ITERATIONS = 6                 # safety limit for the loop

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# 2️⃣  SAMPLE DATASET  — swap for your real data
# ----------------------------------------------------------
agent_df = pd.DataFrame({
    "month":   ["Jan","Feb","Mar","Apr","May","Jun"],
    "revenue": [12000, 15000, 13000, 18000, 22000, 20000],
    "cost":    [8000,  9000,  8500, 11000, 13000, 12000],
    "customers":[120,  145,   132,  175,   210,   195],
})

# ----------------------------------------------------------
# 3️⃣  TOOL: execute_python_code
# ----------------------------------------------------------
_exec_globals = {"df": agent_df, "pd": pd, "plt": plt}

def _run_python(code: str) -> str:
    """Execute code produced by the agent and capture output."""
    import io, sys, traceback
    _buf = io.StringIO()
    sys.stdout = _buf
    try:
        exec(code, _exec_globals)        # share df across calls
        result = _buf.getvalue() or "(code ran, no stdout)"
    except Exception:
        result = traceback.format_exc()
    finally:
        sys.stdout = sys.__stdout__
    return result

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Execute Python code to analyse the DataFrame `df` "
                "or produce matplotlib charts. "
                "Always print() results you want to surface."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# ----------------------------------------------------------
# 4️⃣  SYSTEM PROMPT
# ----------------------------------------------------------
_df_info = (
    f"Columns: {list(agent_df.columns)}\n"
    f"Shape  : {agent_df.shape}\n"
    f"Sample :\n{agent_df.head(3).to_string(index=False)}"
)

SYSTEM_PROMPT = f"""
You are an expert Data Analyst AI Agent.

You have access to a pandas DataFrame called `df`.
{_df_info}

Your job:
1. Understand the user's analytical question.
2. Write concise Python code (using `execute_python_code`) to answer it.
3. Interpret the output and give a clear, insightful answer.
4. If a chart helps, generate it with matplotlib and call plt.savefig('chart.png') then plt.close().

Rules:
- Always print() key numbers / tables so they appear in the tool output.
- Be concise but insightful in your final answer.
- If the first attempt has errors, fix and retry automatically.
""".strip()

# ----------------------------------------------------------
# 5️⃣  AGENT LOOP  (ReAct: Reason → Act → Observe → Repeat)
# ----------------------------------------------------------
def run_analyst_agent(user_question: str) -> str:
    """Run the data analyst agent and return the final answer."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_question},
    ]

    print(f"\n{'='*60}")
    print(f"❓ Question: {user_question}")
    print('='*60)

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        # ── No tool call → agent produced final answer ──────────
        if not msg.tool_calls:
            answer = msg.content or "(no answer)"
            print(f"\n🤖 Agent Answer:\n{textwrap.fill(answer, 80)}")
            return answer

        # ── Process each tool call ───────────────────────────────
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            code   = args["code"]
            print(f"\n🔧 Executing code (iteration {iteration+1}):\n{'-'*40}\n{code}\n{'-'*40}")
            output = _run_python(code)
            print(f"📊 Output:\n{output}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      output,
            })

    return "⚠️  Agent reached maximum iterations without a final answer."

# ----------------------------------------------------------
# 6️⃣  RUN EXAMPLE QUERIES
# ----------------------------------------------------------
#run_analyst_agent("What is the total revenue and profit for each month? Show the top month.")
run_analyst_agent("Plot a bar chart of monthly revenue vs cost and save it.")
#run_analyst_agent("What is the average revenue per customer by month? Which month was most efficient?")
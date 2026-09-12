# -*- coding: utf-8 -*-
"""
作用：手写一遍 Function Calling 全流程 —— 不套 LangChain 的 Agent，直接看底层报文。
效果：亲眼看到 ① 发给模型的 tools 参数长什么样 ② 模型返回的 tool_calls 结构
      ③ 工具结果怎么回传 ④ 五步链路怎么串起来。
"""
import json  # 作用：解析模型给的参数（是 JSON 字符串，不是字典）。
import os  # 作用：读环境变量。
from typing import Any  # 作用：类型注解。

from dotenv import load_dotenv  # 作用：加载 .env。
from openai import OpenAI  # 作用：OpenAI 官方 SDK；效果：DeepSeek 兼容它，改 base_url 即可。

from tools import TOOLS  # 作用：复用第 2 块定义的工具清单。

load_dotenv()  # 作用：注入环境变量。

client = OpenAI(  # 作用：创建客户端；效果：请求打到 DeepSeek。
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("CHAT_MODEL", "deepseek-flash")  # 作用：模型名。

SYSTEM_PROMPT = "你是企业智能助手。需要查资料、算报销、查员工信息时，调用提供的工具；能直接回答就别调工具。"  # 作用：人设；效果：约束调用行为。

TOOL_MAP = {t.name: t for t in TOOLS}  # 作用：名字 → 工具对象；效果：模型说调哪个名字，就按名字找到真函数。

def to_openai_tools(tools: list) -> list[dict]:
    """作用：把 LangChain 工具翻译成 OpenAI 的 tools 参数结构；效果：这就是"声明工具"的那份 JSON Schema。

    参数：
        tools: LangChain 工具列表
    返回：
        list[dict]: 符合 OpenAI 规范的 tools 数组
    """
    return [
        {
            "type": "function",  # 作用：声明这是可调用的函数类型。
            "function": {
                "name": t.name,  # 作用：工具名；效果：模型回调用它。
                "description": t.description,  # 作用：工具说明；效果：模型靠它判断"要不要用这个工具"。
                "parameters": t.args_schema.model_json_schema(),  # 作用：参数结构；效果：模型靠它填对参数（含每个参数的 description）。
            },
        }
        for t in tools
    ]

OPENAI_TOOLS = to_openai_tools(TOOLS)  # 作用：只翻译一次，全局复用。

def run(question: str, max_steps: int = 6, verbose: bool = True) -> str:
    """作用：跑完一次完整的 Function Calling 循环；效果：返回最终答案文本。

    参数：
        question: 用户问题
        max_steps: 最大循环步数；效果：防止模型反复调工具导致死循环
        verbose: 是否打印中间过程
    返回：
        str: 模型最终回答
    """
    messages: list[dict[str, Any]] = [  # 作用：对话历史；效果：每轮都把它整体发出去，模型才知道上下文。
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):  # 作用：ReAct 循环；效果：模型调工具 → 执行 → 回传 → 再想，直到不用工具。
        resp = client.chat.completions.create(  # 作用：发请求；效果：拿到模型这一轮的回复。
            model=MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,  # 作用：把工具清单带上；效果：模型才知道有哪些工具可用。
            tool_choice="auto",  # 作用：让模型自己决定调不调工具（也可强制调用某个）。
        )
        msg = resp.choices[0].message  # 作用：取本轮消息。

        if not msg.tool_calls:  # 作用：模型没要求调工具；效果：说明它要直接回答了。
            if verbose:
                print(f"  [第{step}步] 模型不再调工具，直接作答")
            return msg.content or ""

        messages.append({  # 作用：把"模型想调工具"记进历史；效果：下一轮请求必须带上它，否则模型不知道自己在等结果。
            "role": "assistant",
            "content": msg.content or "",  # 作用：思考模式下 content 可能是空串，这里兜成 ""。
            "tool_calls": [  # 作用：原样保留 tool_calls 结构；效果：这是回传结果时配对用的凭证。
                {
                    "id": c.id,  # 作用：本次调用的唯一编号；效果：回传结果时要靠它配对。
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},  # 作用：名字 + 参数（字符串！）。
                }
                for c in msg.tool_calls
            ],
        })

        for call in msg.tool_calls:  # 作用：一次可能返回多个调用；效果：逐个执行（并行工具调用）。
            name = call.function.name  # 作用：模型说要调哪个工具。
            raw_args = call.function.arguments  # 作用：参数原文；效果：⚠️ 它是 JSON 字符串，不是字典。
            if verbose:
                print(f"  [第{step}步] 模型决定调用 {name}，参数原文：{raw_args}")

            try:
                args = json.loads(raw_args)  # 作用：字符串 → 字典；效果：才能作为 kwargs 传给函数。
            except json.JSONDecodeError:
                result = f"参数不是合法 JSON：{raw_args}"  # 作用：兜住模型给坏参数；效果：把错误回传给它自己纠正。
            else:
                tool = TOOL_MAP.get(name)  # 作用：按名字找工具；效果：防止模型编造一个不存在的工具名。
                if tool is None:
                    result = f"不存在名为 {name} 的工具"
                else:
                    result = str(tool.invoke(args))  # 作用：本地真正执行；效果：模型只"说"，执行的是我们的代码。

            preview = result[:80].replace("\n", " ")  # 作用：截断 + 去换行；效果：日志更好看。
            if verbose:
                print(f"          工具返回：{preview}...")

            messages.append({  # 作用：把结果回传进历史；效果：tool_call_id 必须与上面那条配对，否则接口报错。
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return "达到最大步数仍未得出结果（可能是工具反复失败）"

if __name__ == "__main__":
    print("=" * 64)
    print("① 发给模型的 tools 参数（截取前 1200 字符）—— 这就是 JSON Schema")
    print("=" * 64)
    print(json.dumps(OPENAI_TOOLS, ensure_ascii=False, indent=2)[:1200], "\n...")

    QUESTIONS = [  # 作用：四类问题；效果：分别验证"单个工具""另一个工具""检索""多工具串联"。
        "出差住 3 晚能报多少？",
        "E1002 是谁？",
        "公司报销住宿费需要什么凭证？",
        "公司出差住宿标准是多少？我住 3 晚花了 1500，能报多少？",
    ]

    for q in QUESTIONS:
        print("=" * 64)
        print("问题：", q)
        print("=" * 64)
        answer = run(q)
        print("最终回答：", answer)
        print()
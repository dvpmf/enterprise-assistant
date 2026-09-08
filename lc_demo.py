# -*- coding: utf-8 -*-
"""
作用：用 LangChain 调用本地 Ollama，实现带人设的多轮记忆对话。
效果：运行后在命令行与模型对话，输入 exit 退出；模型能记住前文。
"""
from langchain_ollama import ChatOllama  # 作用：连接 Ollama 的转接头；效果：统一接口调用模型。
from langchain_core.messages import (
                                             SystemMessage,
                                             HumanMessage,
                                             AIMessage,
) # 作用：引入消息对象类；效果：用对象代替字典管理对话。

# 作用：指定模型与关闭思考模式；效果：响应稳定、不把话写进 thinking。
llm = ChatOllama(model="qwen3.5:4b", reasoning=False)

# 作用：定义系统人设；效果：模型按企业助手身份回答。
SYSTEM_PROMPT = ("你是[企业智能助手]，专门帮员工查询企业内部资料、解答工作问题。"
                 "要求：回答简洁专业；不知道就明说不知道，绝不编造。")

# 作用：创建消息列表并放入人设；效果：相当于 chat.py 里 history 的初始 system 消息。
messages = [SystemMessage(content=SYSTEM_PROMPT)]


def main():
    # 作用：提示用户输入并循环对话；效果：实现命令行交互。
    print("企业智能助手已启动（LangChain 版，输入 exit 退出）")
    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            break
        # 作用：把用户输入包装成 HumanMessage 加入列表；效果：记录本轮问题。
        messages.append(HumanMessage(content=user_input))
        # 作用：调用模型获取回复；效果：模型基于全部历史生成回答。
        reply = llm.invoke(messages).content
        # 作用：把模型回答包装成 AIMessage 加入列表；效果：让模型"记住"自己说过的话。
        messages.append(AIMessage(content=reply))
        print("助手：", reply)


if __name__ == "__main__":
    main()
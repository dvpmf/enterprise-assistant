# -*- coding:utf-8 -*- 通过 HTTP 调用本地 Ollama 模型，实现单轮对话。

import requests

# Ollama 聊天接口地址（本机 11434 端口）
OLLAMA_URL = 'http://localhost:11434/api/chat'

MODEL = 'qwen3.5:4b'


def chat_with_model(prompt: str) -> str:
    # 请求体：声明模型、消息、非流式输出
    payload = {
        'model': MODEL,
        'messages':[{
            'role':'user',
            'content':prompt,
            }],
        'stream':False
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["message"]["content"]
if __name__ == '__main__':
    question = '用一句话介绍什么是RAG'
    print('提问：',question)
    print('模型回答：',chat_with_model(question))
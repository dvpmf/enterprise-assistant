#-*- coding: utf-8 -*-


import requests

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'qwen3.5:4b'

SYSTEM_PROMPT = ('你是[企业智能助手]，专门帮员工查询企业内部资料、解答工作问题。'
                 '要求：回答简洁专业；不知道就明说不知道，绝不编造。')

def build_payload(message:list)->dict:#把消息列表组装成发给 Ollama 的请求体。
    return {'model' : MODEL,
            'messages': message,
            'think':False,
            'stream':False}
def ask(message:list)->str:
    resp = requests.post(OLLAMA_URL, json = build_payload(message))
    resp.raise_for_status()
    return resp.json()['message']['content']
def main()->None:
    history = [{'role':'system',
                'content':SYSTEM_PROMPT}]
    print('企业智能助手已启动（输入 exit 退出）')
    while True:
        user_input = input('你：')
        if user_input.strip().lower() =='exit':
            break
        history.append({'role':'user',
                        'content':user_input})
        reply = ask(history)
        history.append({'role':'assistant',
                        'content':reply})
        print('助手:',reply)


if __name__ == '__main__':
    main()
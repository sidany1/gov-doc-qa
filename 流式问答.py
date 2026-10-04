import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-cef624d559fd4a6fb2d6774989fbbb91",
    base_url="https://api.deepseek.com")

def appText(messages,content):
    text = {"role": "user", 'content': content}
    messages.append(text)
    return messages

def appPrintText(messages,aiContent):
    printText = {"role": "assistant", 'content': aiContent}
    messages.append(printText)
    return messages


if __name__ == '__main__':
    message = [
        {"role": "system", "content": "你是一个有用的助手，可以帮助我完成一些任务。"},
    ]
    content = ""
    while (True):
        content = input('请输入问题:')
        if(content == "退出"):
            break
        message = appText(message,content)

        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = message,
            stream=True
        )
        aiContent =""
        for i in response:
            if i.choices[0].delta.content is not None:
                aiContent += i.choices[0].delta.content
                print(i.choices[0].delta.content,end="")
        print()
        message = appPrintText(message, aiContent)
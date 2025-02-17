# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import streamlit as st
from Agent.ReAct import ReActAgent
from Models.Factory import ChatModelFactory
from Tools import *
from Tools.PythonTool import ExcelAnalyser
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory


st.set_page_config(page_title="欢迎来到IAI李剑的网站，请提出你的问题")
st.title("欢迎来到IAI李剑的网站，请提出你的问题")

model_id = "MateConv"

def init_chat_messages():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("Hello~我是IAI李剑团队开发的MateConv，很高兴为您服务😄")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages

def launch_agent(agent: ReActAgent):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    chat_history = ChatMessageHistory()

    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, chat_history, verbose=True)
        print(f"{ai_icon}：{reply}\n")

def main():


    messages = init_chat_messages()


    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()

            chat_messages = []
            chat_messages.append({"role": "user", "content": prompt})


    # 语言模型
    llm = ChatModelFactory.get_model("gpt-4o-2024-05-13")
    #llm = ChatModelFactory.get_model("deepseek")

    # 自定义工具集
    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(
            llm=llm,
            prompt_file="F:\AI\learning\lecture-notes\/10-agent\/autogpt\/auto-gpt-work\prompts\/tools\excel_analyser.txt",
            verbose=True
        ).as_tool()
    ]

    # 定义智能体
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        work_dir="F:\AI\learning\lecture-notes\/10-agent\/autogpt\/auto-gpt-work\data",
        main_prompt_file="F:\AI\learning\lecture-notes\/10-agent\/autogpt\/auto-gpt-work\prompts\main\main.txt",
        max_thought_steps=5,
    )

    # 运行智能体
    launch_agent(agent)


if __name__ == "__main__":
    main()

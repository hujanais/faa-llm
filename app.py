from agents.chat_agent import ChatAgent

if __name__ == "__main__":
    agent = ChatAgent()

    while True:
        user_input = input(">> ")
        if user_input.lower() == "bye":
            print("LLM: Goodbye")
            break

        if user_input is not None:
            print(agent.chat(user_input))

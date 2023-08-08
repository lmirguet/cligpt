# an implementation for a terminal-based ChatGPT
#
import openai
import configparser
from termcolor import colored
import argparse
import os
import shutil

USER = "user"
SYSTEM = "system"
ASSISTANT = "assistant"

class Message:
    def __init__(self, role, message) -> None:
        if role not in [ USER, SYSTEM, ASSISTANT ]:
            raise ValueError("The role can only one of 'system', 'user' or 'assitant'")

        self.role = role
        self.content = message
    
def format_message(message: Message):
    return { "role": message.role, "content": message.content }

class ChatSession:
    def __init__(self, openai_model: str, openai_temperature: float, assistant_color: str) -> None:
        self.model = openai_model
        self.messages = []
        self.temperature = openai_temperature
        self.color = assistant_color

    def add_message(self, message: Message):
        self.messages.append(message)

    def call_streamed_chat_completion(self):
        chat_completion = openai.ChatCompletion.create(
                model = self.model,
                messages = list(map(format_message, self.messages)),
                temperature = self.temperature,
                stream = True
        )
        try:
            for chunk in chat_completion:
                current_content = chunk["choices"][0]["delta"].get("content","")
                yield current_content
        except Exception as e:
            print("OpenAI Response (Streaming) Error: "+str(e))

    def print_and_compile_message(self, response):
        result = ""
        print()
        for c in response:
            result += c
            print(colored(c, self.color), end="")
        print()
        print()
        return Message(ASSISTANT, result)

    def interactive_session(self):
        while (True):
            user_input = input("> ")
            if user_input.upper() in [ "BYE", "STOP", "QUIT" ]:
                break
            user_message = Message(USER, user_input)
            self.add_message(user_message)
            response = self.call_streamed_chat_completion()
            response_message = self.print_and_compile_message(response)
            self.add_message(response_message)

def list_models():
    models = openai.Model.list()
    for i in models.data:
        if i.id.startswith("gpt-3.5") or i.id.startswith("gpt-4"):
            print(i.id)

def check_if_config_exists_and_api_key() -> configparser.ConfigParser:
    setting_file = "config.ini"
    config_modified = False

    if not os.path.exists(setting_file):
        shutil.copy("config_sample.ini", setting_file)

    config = configparser.ConfigParser()
    config.read(setting_file)

    openai_key = config["settings"]["OPENAI_API_KEY"]

    if len(openai_key) < 5:
        openai_key = input("Enter your OpenAI secret key: ")
        config["settings"]["OPENAI_API_KEY"] = openai_key
        config_modified = True

    openai.api_key = openai_key

    try:
        openai.Engine.list()
    except openai.error.OpenAIError as e:
        if "authentication" in str(e).lower():
            print("This openai API key is invalid.")
            exit(1)
        raise  # If the error is due to some other reason, raise it

    if config_modified:
        with open(setting_file, "w") as fp:
            config.write(fp)

    return config

def main():
    parser = argparse.ArgumentParser(description="A CLI for ChatGPT")
    parser.add_argument("--list-models", action="store_true", help="List the available OpenAI models")
    parser.add_argument("--set-model", type=str, help="Set a certain model to be used with cligpt")
    args = parser.parse_args()

    config = check_if_config_exists_and_api_key()

    openai_model = config["settings"]["OPENAI_MODEL"]
    openai_temperature = float(config["settings"]["OPENAI_TEMPERATURE"])
    assistant_color = config["settings"]["ASSISTANT_COLOR"]

    if args.list_models == True:
        list_models()
        exit(0)

    if args.set_model is not None:
        with open("config.ini", "w") as fp:
            config["settings"]["OPENAI_MODEL"] = args.set_model
            config.write(fp)
        exit(0)

    # run chat session
    session = ChatSession(openai_model, openai_temperature, assistant_color)
    session.interactive_session()

if __name__ == "__main__":
    main()


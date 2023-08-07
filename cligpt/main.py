# an implementation for a terminal-based ChatGPT
#
import openai
import configparser
from termcolor import colored
import argparse

USER = "user"
SYSTEM = "system"
ASSISTANT = "assistant"

class Message:
    def __init__(self, role, message) -> None:
        if role not in [ USER, SYSTEM, ASSISTANT ]:
            raise ValueError("The role can only one of 'system', 'user' or 'assitant'")

        self.role = role
        self.content = message

    def print(self, assistant_color):
        print()
        if self.role == ASSISTANT:
            print(colored(self.content, assistant_color))
        else:
            print(self.content)
        print()
    
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

    def interactive_session(self):
        while (True):
            user_input = input("> ")
            if user_input.upper() in [ "BYE", "STOP", "QUIT" ]:
                break
            user_message = Message(USER, user_input)
            self.add_message(user_message)
            chat_completion = openai.ChatCompletion.create(
                model = self.model,
                messages = list(map(format_message, self.messages)),
                temperature = self.temperature
            )
            completion_obj = chat_completion.choices[0]
            response_message = Message(completion_obj['message']['role'],
                                    completion_obj['message']['content'])
            self.add_message(response_message)
            response_message.print(self.color)

def list_models():
    models = openai.Model.list()
    for i in models.data:
        print(i.id)

def main():
    parser = argparse.ArgumentParser(description="A CLI for ChatGPT")
    parser.add_argument("--list-models", action="store_true", help="List the available OpenAI models")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("config.ini")

    openai.api_key = config["settings"]["OPENAI_API_KEY"]
    openai_model = config["settings"]["OPENAI_MODEL"]
    openai_temperature = float(config["settings"]["OPENAI_TEMPERATURE"])
    assistant_color = config["settings"]["ASSISTANT_COLOR"]

    if args.list_models == True:
        list_models()
        exit(0)

    # run chat session
    session = ChatSession(openai_model, openai_temperature, assistant_color)
    session.interactive_session()

if __name__ == "__main__":
    main()


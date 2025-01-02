#
# An implementation for a terminal-based ChatGPT
# Apache License (2.0)
#
import openai
from openai import OpenAI

import configparser
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.key_binding import KeyBindings

import argparse
import os
import shutil

USER = "user"
SYSTEM = "system"
ASSISTANT = "assistant"
IMAGE_QUALITY = [ "standard", "hd" ]

class Message:
    """
    Represents a chat message with attributes like role and content
    """

    def __init__(self, role, message) -> None:
        if role not in [USER, SYSTEM, ASSISTANT]:
            raise ValueError("The role can only one of 'system', 'user' or 'assitant'")

        self.role = role
        self.content = message


def format_message(message: Message):
    """
    A utility function to transform a message into a dict
    (useful for mapping a list of messages)
    """
    return {"role": message.role, "content": message.content}


class ChatSession:
    """
    Represents an interactive chat session with OpenAI's ChatGPT.
    This class has methods to add messages, call OpenAI's chat completion
    endpoint, print responses, and run an interactive session
    """

    def __init__(
        self, openai_model: str, openai_temperature: float, assistant_color: str, client: OpenAI
    ) -> None:
        self.model = openai_model
        self.messages = []
        self.temperature = openai_temperature
        self.color = assistant_color
        self.client = client

    def add_message(self, message: Message):
        self.messages.append(message)

    def call_streamed_chat_completion(self):
        chat_completion = self.client.chat.completions.create(model=self.model,
            messages=list(map(format_message, self.messages)),
            temperature=self.temperature,
            stream=True)
        try:
            for chunk in chat_completion:
                current_content = chunk.choices[0]
                yield current_content
        except Exception as e:
            print("OpenAI Response (Streaming) Error: " + str(e))

    def print_colored_text(self, text, color):
            formatted_text = FormattedText([(color, text)])
            print_formatted_text(formatted_text, end='')

    def print_and_compile_message(self, response):
        result = ""
        max_token_exceeded = False
        print()
        for c in response:
            if c.finish_reason is not None and c.finish_reason == "length":
                max_token_exceeded = True
            else:
                txt = c.delta.content
                if txt is not None:
                    result += txt
                    self.print_colored_text(txt, self.color)

        print()
        print()
        if max_token_exceeded:
            print("Maximum number of tokens exceeded.")
            exit(0)
        return Message(ASSISTANT, result)

    def interactive_session(self, first_message=None, current_directory=None):
        session = PromptSession()
        kb = KeyBindings()

        @kb.add('c-space')
        def _(event):
            event.app.current_buffer.validate_and_handle()

        @kb.add('enter')
        def _(event):
            event.app.current_buffer.insert_text('\n')

        if first_message is not None:
            self.session_instance(first_message)

        with patch_stdout():
            while True:
                try:
                    user_input = session.prompt("> ", key_bindings=kb, multiline=True)
                except (EOFError, KeyboardInterrupt):
                    break

                if user_input.upper() in ["BYE", "STOP", "QUIT", "Q"]:
                    break
                if user_input.upper() in ("S", "START", "START_PROMPT"):  # multi-line prompts
                    user_input = ""
                    while True:
                        try:
                            inp = session.prompt("> ", key_bindings=kb, multiline=True)
                        except (EOFError, KeyboardInterrupt):
                            break
                        if inp.upper() in ("E", "END", "END_PROMPT"):
                            break
                        inp = self.check_if_load_file(current_directory, inp)
                        user_input += inp + '\n'
                user_input = self.check_if_load_file(current_directory, user_input)

                if self.check_if_requesting_image(user_input):
                    continue
                if self.check_if_requesting_imageedit(user_input, current_directory):
                    continue

                self.session_instance(user_input)

    def check_if_requesting_imageedit(self, user_input, current_directory):
        sub_input = user_input.split()
        if len(sub_input) < 3:
            return False
        if sub_input[0].upper() in [ "IMGEDIT" ]:
            filename = sub_input[1]
            filepath = os.path.join(current_directory, filename)
            prompt = " ".join(sub_input[2:])
            print("Generating image edit...")
            response = self.client.images.edit(
                image=open(filepath, "rb"),
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            print(response.data[0].url)
            return True
        return False

    def check_if_requesting_image(self, user_input):
        if len(user_input) < 2:
            return False
        ## split user_input into a list of string separated by blanks
        sub_input = user_input.split()
        if sub_input[0].upper() in [ "IMG", "IMAGE"]:
            words_used = 1
            # get the substring of user_input without the first word
            print("Generating image...")
            size = "1792x1024"
            if sub_input[words_used].upper() in IMAGE_QUALITY:
                quality = sub_input[words_used].lower()
                words_used = words_used + 1
            else:
                quality = "standard"
            model = "dall-e-3"
            style = "natural"
            prompt = " ".join(sub_input[words_used:])
            images = self.client.images.generate(prompt=prompt, n=1, size=size, quality=quality, model=model, style=style)
            print(images.data[0].url)
            return True
        return False

    def check_if_load_file(self, current_directory, user_input):
        if user_input.upper().startswith("F "):
            filename = user_input.split(' ')[1]
            if current_directory is not None:
                filepath = os.path.join(current_directory, filename)
            else:
                filepath = filename
            with open(filepath, "r") as fp:
                user_input = fp.read()
        return user_input

    def session_instance(self, user_input):
        user_message = Message(USER, user_input)
        self.add_message(user_message)
        response = self.call_streamed_chat_completion()
        response_message = self.print_and_compile_message(response)
        self.add_message(response_message)


def list_models(client: OpenAI):
    """
    Action for listing the available chat models
    """
    models = client.models.list()
    printedmodels = []
    for i in models.data:
        if i.id.startswith("gpt") or i.id.startswith("chatgpt") or i.id.startswith("o1"):
            printedmodels.append(i.id)
    printedmodels.sort()
    for i in printedmodels:
        print(i)


def get_model(config: configparser.ConfigParser):
    """
    Action for getting the current chat model
    """
    print(config["settings"]["OPENAI_MODEL"])


def check_if_config_exists_and_api_key() -> (configparser.ConfigParser, OpenAI):
    """
    Initialize the configuration, create it if it doesn't exist
    """
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

    client = OpenAI(api_key=openai_key)

    try:
        client.models.list()
    except openai.OpenAIError as e:
        if "authentication" in str(e).lower():
            print("This openai API key is invalid. Or OpenAI API could not be reached.")
            exit(1)
        raise  # If the error is due to some other reason, raise it

    if config_modified:
        with open(setting_file, "w") as fp:
            config.write(fp)

    return config, client


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="""A CLI for ChatGPT.
You can type any question you want to ChatGPT.
Use CTRL+SPACE to validate your input.
Use a simple 'bye', 'stop', 'quit' or 'q' to quit the session (or CTRL+D).
Use 's' or 'start' to start a multi-line prompt, and 'e' or 'end' to end it.
Use 'f FILENAME' to inline a file.
Use 'img' or 'image' to generate an image.
Use 'imgedit' to edit an image.
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--list-models", action="store_true", help="list the available OpenAI models"
    )
    parser.add_argument(
        "--set-model", type=str, help="set a certain model to be used with cligpt"
    )
    parser.add_argument(
        "--get-model", action="store_true", help="get the currently used OpenAI model"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="an input file containing a first message for ChatGPT",
    )
    parser.add_argument(
        "--working-directory",
        type=str,
        help="the working directory from where the files should be loaded"
    )
    args = parser.parse_args()

    config, client = check_if_config_exists_and_api_key()

    openai_model = config["settings"]["OPENAI_MODEL"]
    openai_temperature = float(config["settings"]["OPENAI_TEMPERATURE"])
    assistant_color = config["settings"]["ASSISTANT_COLOR"]

    if args.list_models == True:
        list_models(client)
        exit(0)

    if args.set_model is not None:
        with open("config.ini", "w") as fp:
            config["settings"]["OPENAI_MODEL"] = args.set_model
            config.write(fp)
        exit(0)

    if args.get_model == True:
        get_model(config)
        exit(0)

    first_message = None
    if args.input_file is not None:
        if args.working_directory is not None:
            filepath = os.path.join(args.working_directory, args.input_file)
        else:
            filepath = args.input_file
        with open(filepath, "r") as fp:
            first_message = fp.read()

    # run chat session
    session = ChatSession(openai_model, openai_temperature, assistant_color, client)
    session.interactive_session(first_message, args.working_directory)


if __name__ == "__main__":
    main()

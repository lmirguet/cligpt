from openai import OpenAI
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

openai_key = config["settings"]["OPENAI_API_KEY"]

if len(openai_key) < 5:
    openai_key = input("Enter your OpenAI secret key: ")
    config["settings"]["OPENAI_API_KEY"] = openai_key
    config_modified = True

client = OpenAI(api_key=openai_key)

audio_file = open("alstom3.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  language="en"
)

print(transcript)

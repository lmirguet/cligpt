# cligpt

A terminal-based ChatGPT.

## Requirements

To run this program, you will need:
* a working python environment
* an **OpenAI API key** - once you have your OpenAI account, you can create your own key at https://platform.openai.com/account/api-keys

### Install a virtual environment and launch the program

We recommend that you install the needed python libraries in a virtual environment. Here is how you can do:

Create a virtual environment in the folder where you cloned the projet:
```
python -m venv .venv
```
Then activate it and download the required libraries:
```
.venv/bin/activate
pip install -r requirements.txt
```
Now you can run the program:
```
python cligpt/main.py
```

## OpenAI documentation

see https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb for information

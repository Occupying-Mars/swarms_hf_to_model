import time 
import openai
import json
import os
import requests
from bs4 import BeautifulSoup


api_keys= ""

def asst_code_function(assistant_id, prompt, new_file, file, update):
    assistant_id = assistant_id
    # Define the user prompt
    user_prompt = prompt
    # Assuming 'openapi_schema.yaml' is the name of the file containing the API schema
    # First, upload the file to OpenAI
    client = openai.OpenAI(api_key=api_keys)

    file = client.files.create(file=open(file, "rb"), purpose='assistants')

    # Create a thread with the assistant
    thread = client.beta.threads.create()
    print("https://platform.openai.com/playground?assistant="+assistant_id+"&mode=assistant&thread="+thread.id)
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_prompt,
        file_ids=[file.id]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    time.sleep(120)
    response = client.beta.threads.messages.retrieve(
        thread_id=thread.id,
        message_id=message.id
    )
    messages_response = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    # Parse the messages to find the file ID from the assistant's response
    file_id = None
    for message in messages_response.data:
        if message.role == 'assistant' and message.file_ids:
            files_id = message.file_ids[0]
            break

    print("File ID:", files_id)

    image_data = client.files.content(files_id)
    image_data_bytes = image_data.read()

    mode = 'w' if update == 'overwrite' else 'a'
    with open(new_file, mode) as file:
        file.write(image_data_bytes.decode('utf-8'))

def asst_message_function(assistant_id, prompt,file):
    assistant_id = assistant_id
    # Define the user prompt
    user_prompt = prompt
    # Assuming 'openapi_schema.yaml' is the name of the file containing the API schema
    # First, upload the file to OpenAI
    client = openai.OpenAI(api_key=api_keys)

    file = client.files.create(file=open(file, "rb"), purpose='assistants')

    # Create a thread with the assistant
    thread = client.beta.threads.create()
    print("https://platform.openai.com/playground?assistant="+assistant_id+"&mode=assistant&thread="+thread.id)
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_prompt,
        file_ids=[file.id]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    time.sleep(100)
    response = client.beta.threads.messages.retrieve(
        thread_id=thread.id,
        message_id=message.id
    )
    messages_response = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    response = messages_response.data[0].content[0].text.value
    return response

def create_assistant(assistant_name, file=None):
    client = openai.OpenAI(api_key=api_keys)
    assistant_params = {
        "name": assistant_name,
        "description": "You are a master assistant that can add new models to swarms code you are given a refrence file for the model that is how you add a model to swarms you will also be given contents of a webpage for the specific model you have to add use that add the model accordingly. once done save the code in a file and return the file",
        "model": "gpt-4o"
    }
    if file:
        file = client.files.create(
            file=open(file, "rb"),
            purpose='assistants'
        )
        assistant_params["file_ids"] = [file.id]
    assistant = client.beta.assistants.create(**assistant_params)
    return assistant.id

def scrape_webpage(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all text content
        text_content = soup.get_text(separator=' ', strip=True)

        return text_content

    except requests.RequestException as e:
        print(f"An error occurred while fetching the webpage: {e}")
        return None
#create an assistant
#scrape the webpage url for the model
#upload the scraped contents to assistant and run the assistant
#save the output file

assistant_id = create_assistant("add_new_model" , "llava.py")

url = "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
content = scrape_webpage(url)
if content:
    try:
        with open("model_content.txt", "w", encoding="utf-8") as file:
            file.write(content)
        print("Content saved successfully to 'model_content.txt'")
        
        user_prompt = f"Please add a new model to the swarms library based on the content in 'model_content.txt' and the reference file 'llava.py'. Save the new model implementation in a file called 'newmodel.py'."
        response = asst_code_function(assistant_id, user_prompt, "model_content.txt")
        
        with open("newmodel.py", "w", encoding="utf-8") as output_file:
            output_file.write(response)
        
        print("New model implementation saved in 'newmodel.py'")
    except IOError as e:
        print(f"Error writing to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
else:
    print("No content to save.")

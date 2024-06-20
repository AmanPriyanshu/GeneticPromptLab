import json
import time

def send_query2gpt(client, messages, function_template, temperature=0, pause=5):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=512,
        functions=[function_template], 
        seed=0,
        function_call={"name": function_template["name"]}
    )
    answer = response.choices[0].message.function_call.arguments
    generated_response = json.loads(answer)
    time.sleep(pause)
    return generated_response
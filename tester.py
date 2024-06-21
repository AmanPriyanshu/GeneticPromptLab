from GeneticPromptLab.function_templates import function_templates
import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import trange
from openai import OpenAI

qa_function_template = function_templates[1]
with open("openai_api.key", "r") as f:
    key = f.read()
client = OpenAI(api_key=key.strip())

def read_data(path_test_df, path_label_dict):
    with open(path_label_dict, "r") as f:
        label_dict = json.load(f)
    df = pd.read_csv(path_test_df)
    questions = df['question'].tolist()
    answers = [label_dict[v] for v in df['label'].tolist()]
    return questions, answers, label_dict

def read_latest_epoch_data(run_id):
    dir_path = f"./runs/{run_id}/"
    files = [f for f in os.listdir(dir_path) if f.startswith('epoch_') and f.endswith('.csv')]
    # Sort files based on numerical value of epoch_id
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_file_path = os.path.join(dir_path, sorted_files[-1])
    df = pd.read_csv(latest_file_path)
    return df

def get_highest_fitness_prompt(df):
    max_fitness_row = df[df['Fitness Score'] == df['Fitness Score'].max()]
    highest_fitness_prompt = max_fitness_row['Prompt'].iloc[0]
    return highest_fitness_prompt

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

def ag_news():
    run_id = "XrFnn68pnF"
    path_test_df = "./data/ag_news_test.csv"
    path_label_dict = "./data/ag_news_label_dict.json"
    questions, answers, label_vocab = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)
    best_prompt = get_highest_fitness_prompt(read_latest_epoch_data(run_id))
    batch_size = 10
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = label_vocab
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str(label_vocab)
    qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = batch_size
    qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = batch_size
    aggregate_accuracy = []
    batches_skipped_count = 0
    for i in trange(0, len(questions), batch_size, desc="testing_agnews"):
        question_subset = questions[i:i+batch_size]
        answer_subset = answers[i:i+batch_size]
        questions_list = "\n\n".join([str(i+1)+'. """'+question+'"""' for i,question in enumerate(question_subset)])
        try:
            response = [v['label'] for v in send_query2gpt(client, [{"role": "system", "content": best_prompt}, {"role": "user", "content": "Questions:\n"+questions_list}], qa_function_template, temperature=0.0, pause=5)['label_array']]
            accuracy = sum(1 if a == b else 0 for a, b in zip(response, answer_subset)) / len(response)
            aggregate_accuracy.append(accuracy)
        except:
            batches_skipped_count += 1
    print("Batches skipped", batches_skipped_count)
    print("Accuracy:", str(round(100*np.mean(accuracy), 3))+"%")

def trec():
    run_id = "08zLX4cd97"
    path_test_df = "./data/trec_test.csv"
    path_label_dict = "./data/trec_label_dict.json"
    questions, answers, label_vocab = read_data(path_test_df=path_test_df, path_label_dict=path_label_dict)
    best_prompt = get_highest_fitness_prompt(read_latest_epoch_data(run_id))
    batch_size = 10
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = label_vocab
    qa_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str(label_vocab)
    qa_function_template["parameters"]["properties"]["label_array"]["minItems"] = batch_size
    qa_function_template["parameters"]["properties"]["label_array"]["maxItems"] = batch_size
    aggregate_accuracy = []
    batches_skipped_count = 0
    for i in trange(0, len(questions), batch_size, desc="testing_agnews"):
        question_subset = questions[i:i+batch_size]
        answer_subset = answers[i:i+batch_size]
        questions_list = "\n\n".join([str(i+1)+'. """'+question+'"""' for i,question in enumerate(question_subset)])
        try:
            response = [v['label'] for v in send_query2gpt(client, [{"role": "system", "content": best_prompt}, {"role": "user", "content": "Questions:\n"+questions_list}], qa_function_template, temperature=0.0, pause=5)['label_array']]
            accuracy = sum(1 if a == b else 0 for a, b in zip(response, answer_subset)) / len(response)
            aggregate_accuracy.append(accuracy)
        except:
            batches_skipped_count += 1
    print("Batches skipped", batches_skipped_count)
    print("Accuracy:", str(round(100*np.mean(aggregate_accuracy), 3))+"%")

def main():
    ag_news()
    # trec()

if __name__=='__main__':
    main()
from openai import OpenAI
import pandas as pd
import json
from GeneticPromptLab import QuestionsAnswersOptimizer

with open("openai_api.key", "r") as f:
    key = f.read()
client = OpenAI(api_key=key.strip())

def trec():
    # Configuration
    train_path = './data/trec_train.csv'
    test_path = './data/trec_test.csv'
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    with open("./data/trec_label_dict.json", "r") as f:
        label_dict = json.load(f)
        label_dict = {i:v for i,v in enumerate(label_dict)}
    problem_description = "Data are collected from four sources: 4,500 English questions. Your objective is to classify these into one of the following labels: "+str(label_dict)

    train_questions_list, train_answers_label, test_questions_list, test_answers_label = train_data['question'].tolist(), train_data['label'].tolist(), test_data['question'].tolist(), test_data['label'].tolist()
    # Create GeneticPromptLab instance
    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name

def agnews():
    # Configuration
    train_path = './data/ag_news_train.csv'
    test_path = './data/ag_news_test.csv'
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    with open("./data/ag_news_label_dict.json", "r") as f:
        label_dict = json.load(f)
        label_dict = {i:v for i,v in enumerate(label_dict)}
    problem_description = "AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining. Your objective is a classification label, with possible values including World (0), Sports (1), Business (2), Sci/Tech (3)."

    train_questions_list, train_answers_label, test_questions_list, test_answers_label = train_data['question'].tolist(), train_data['label'].tolist(), test_data['question'].tolist(), test_data['label'].tolist()
    # Create GeneticPromptLab instance
    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name

def main():
    print("AGNEWS:")
    problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name = agnews()
    population_size = 8
    generations = 10
    sample_p = 0.01
    num_retries = 2

    lab = QuestionsAnswersOptimizer(
        client=client, 
        problem_description=problem_description, 
        train_questions_list=train_questions_list, 
        train_answers_label=train_answers_label, 
        test_questions_list=test_questions_list, 
        test_answers_label=test_answers_label, 
        label_dict=label_dict, 
        model_name=model_name, 
        sample_p=sample_p, 
        init_and_fitness_sample=population_size, 
        window_size_init=2,
        num_retries=num_retries)
    optimized_prompts = lab.genetic_algorithm(generations)
    print(optimized_prompts)
    print("-------- EXPERIMENT COMPLETED --------")
    print("TREC:")
    problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name = trec()
    population_size = 8
    generations = 10
    sample_p = 0.01
    num_retries = 2


    lab = QuestionsAnswersOptimizer(
        client=client, 
        problem_description=problem_description, 
        train_questions_list=train_questions_list, 
        train_answers_label=train_answers_label, 
        test_questions_list=test_questions_list, 
        test_answers_label=test_answers_label, 
        label_dict=label_dict, 
        model_name=model_name, 
        sample_p=sample_p, 
        init_and_fitness_sample=population_size, 
        window_size_init=2, 
        num_retries=num_retries)
    optimized_prompts = lab.genetic_algorithm(generations)
    print(optimized_prompts)
    print("-------- EXPERIMENT COMPLETED --------")

if __name__=='__main__':
    main()
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from openai import OpenAI
from function_templates import function_templates

with open("openai_api.key", "r") as f:
    key = f.read()
client = OpenAI(api_key=key.strip())

def send_query2gpt(client, messages, function_template):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=512,
        functions=[function_template], 
        function_call={"name": function_template["name"]}
    )
    answer = response.choices[0].message.function_call.arguments
    generated_response = json.loads(answer)
    return generated_response

class GeneticPromptLab:
    def __init__(self, problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name, sample_p=1.0, init_and_fitness_sample=10, window_size_init=1):
        self.init_and_fitness_sample = init_and_fitness_sample
        self.test_questions_list = test_questions_list
        self.test_answers_label = test_answers_label
        self.label_dict = label_dict
        self.problem_description = problem_description
        self.window_size_init = window_size_init

        self.model = SentenceTransformer(model_name)
        self.sample_p = sample_p
        train_indices_list = np.random.choice(np.arange(len(train_questions_list)), size=int(len(train_questions_list)*self.sample_p))
        self.train_questions_list = [train_questions_list[i] for i in train_indices_list]
        self.train_answers_label = [train_answers_label[i] for i in train_indices_list]
        self.embeddings = self.model.encode(self.train_questions_list, show_progress_bar=True)
        self.already_sampled_indices = set()

    def create_prompts(self, data):
        data_doubled = data+data
        batched_messages = []
        for i in range(len(data)):
            sample = data_doubled[i:i+self.window_size_init]
            sample_prompt = "\n".join(["Question: \"\"\""+s["q"]+"\"\"\"\nCorrect Label:\"\"\""+s["a"]+"\"\"\"" for s in sample])
            messages = [{"role": "system", "content": function_templates[0]["description"]+"\n\nNote: For this task the labels are: "+"\n".join([str(k)+". "+str(v) for k,v in self.label_dict.items()])}, {"role": "user", "content": "Observe the following samples:\n\n"+sample_prompt}]
            batched_messages.append(messages)
        return batched_messages

    def generate_init_prompts(self):
        distinct_samples = self.sample_distinct(self.init_and_fitness_sample)
        data = []
        for sample_index in distinct_samples:
            question = self.train_questions_list[int(sample_index)]
            answer = self.train_answers_label[int(sample_index)]
            data.append({"q": question, "a": self.label_dict[answer]})
        prompts = self.create_prompts(data)
        
        return prompts

    def sample_distinct(self, n):
        embeddings = self.embeddings

        if len(self.already_sampled_indices) > 0:
            mask = np.ones(len(embeddings), dtype=bool)
            mask[list(self.already_sampled_indices)] = False
            embeddings = embeddings[mask]

        kmeans = KMeans(n_clusters=n, random_state=0).fit(embeddings)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        sampled_indices = set(closest_indices)

        while len(sampled_indices) < n:
            remaining_indices = set(range(len(embeddings))) - sampled_indices
            remaining_embeddings = embeddings[list(remaining_indices)]
            kmeans = KMeans(n_clusters=n - len(sampled_indices), random_state=0).fit(remaining_embeddings)
            _, closest_indices = pairwise_distances_argmin_min(kmeans.cluster_centers_, remaining_embeddings)
            sampled_indices.update(closest_indices)

        sampled_indices = list(sampled_indices)[:n]
        self.already_sampled_indices.update(sampled_indices)
        return sampled_indices

    def genetic_algorithm(self, generations, mutation_rate=0.1):
        initial_prompts = self.generate_init_prompts()
        population = initial_prompts

        for _ in range(generations):
            fitness_scores = self.evaluate_fitness(population)
            top_prompts = self.select_top_prompts(fitness_scores)

            new_prompts = self.crossover_using_gpt(top_prompts)

            num_random_prompts = int(population_size * 0.25)
            random_prompts = self.random_initialization(num_random_prompts)

            population = new_prompts + random_prompts
            population = self.mutate_prompts(population, mutation_rate)

        return population

    def evaluate_fitness(self, prompts):
        return {prompt: random.uniform(0, 1) for prompt in prompts}

    def select_top_prompts(self, fitness_scores, top_fraction=0.5):
        sorted_prompts = sorted(fitness_scores, key=fitness_scores.get, reverse=True)
        cutoff = int(len(sorted_prompts) * top_fraction)
        return sorted_prompts[:cutoff]

    def crossover_using_gpt(self, prompts):
        new_prompts = []
        for i in range(0, len(prompts), 2):
            if i + 1 < len(prompts):
                template = prompts[i]
                additive = prompts[i + 1]
                new_prompt = gpt_mix_and_match(template, additive)
                new_prompts.append(new_prompt)
        return new_prompts

    def random_initialization(self, size):
        return [f"Random initialized prompt {_}" for _ in range(size)]

    def mutate_prompts(self, prompts, mutation_rate=0.1):
        mutated_prompts = []
        for prompt in prompts:
            if random.random() < mutation_rate:
                mutated_prompts.append(gpt_mutate(prompt))
            else:
                mutated_prompts.append(prompt)
        return mutated_prompts

if __name__ == '__main__':
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

    population_size = 10
    generations = 10
    sample_p = 0.01

    lab = GeneticPromptLab(problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name, sample_p, init_and_fitness_sample=population_size)
    optimized_prompts = lab.genetic_algorithm(generations)

    print("Optimized Prompts:", optimized_prompts)

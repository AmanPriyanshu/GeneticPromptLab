import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from openai import OpenAI

with open("openai_api.key", "r") as f:
    key = f.read()
client = OpenAI(api_key=key.strip())

def send_query2gpt(client, messages, function_template):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=512,
        functions=function_template, 
        function_call={"name": function_template[0]["name"]}
    )
    answer = response.choices[0].message.function_call.arguments
    generated_response = json.loads(answer)
    return generated_response

class GeneticPromptLab:
    def __init__(self, train_path, test_path, model_name, sample_p=1.0, init_and_fitness_sample=10):
        self.init_and_fitness_sample = init_and_fitness_sample
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.model = SentenceTransformer(model_name)
        self.sample_p = sample_p
        self.train_data = self.train_data.sample(frac=self.sample_p)
        self.embeddings = self.model.encode(self.train_data['question'].tolist(), show_progress_bar=True)
        self.already_sampled_indices = set()

    def generate_init_prompts(self, N):
        subset_n = N//self.init_and_fitness_sample
        distinct_samples = self.sample_distinct(self.init_and_fitness_sample)
        ## FILL HERE

    def sample_distinct(self, n):
        num_samples = int(len(self.train_data) * self.sample_p)
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

    def genetic_algorithm(self, population_size, generations, mutation_rate=0.1):
        initial_prompts = self.generate_init_prompts(population_size)
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
    sample_p = 0.01

    # Create GeneticPromptLab instance
    lab = GeneticPromptLab(train_path, test_path, model_name, sample_p)
    population_size = 100
    generations = 10
    optimized_prompts = lab.genetic_algorithm(population_size, generations)

    print("Optimized Prompts:", optimized_prompts)

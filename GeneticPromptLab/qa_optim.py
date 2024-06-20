import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import string
from tqdm import tqdm
from .utils import send_query2gpt
from .function_templates import function_templates
import warnings
from .base_class import GeneticPromptLab
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QuestionsAnswersOptimizer(GeneticPromptLab):
    def __init__(self, client, problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name, sample_p=1.0, init_and_fitness_sample=10, window_size_init=1, generations=10, num_retries=1):
        self.num_retries = num_retries
        self.client = client
        self.generations = generations
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
        prompts = []
        for i in range(len(data)):
            sample = data_doubled[i:i+self.window_size_init]
            sample_prompt = "\n".join(["Question: \"\"\""+s["q"]+"\"\"\"\nCorrect Label:\"\"\""+s["a"]+"\"\"\"" for s in sample])
            messages = [{"role": "system", "content": "Problem Description: "+self.problem_description+"\n\n"+function_templates[0]["description"]+"\n\nNote: For this task the labels are: "+"\n".join([str(k)+". "+str(v) for k,v in self.label_dict.items()])}, {"role": "user", "content": "Observe the following samples:\n\n"+sample_prompt}]
            prompt = send_query2gpt(self.client, messages, function_templates[0])['prompt']
            prompts.append(prompt)
        return prompts

    def generate_init_prompts(self, n=None):
        if n is None:
            n = self.init_and_fitness_sample
        distinct_sample_indices = self.sample_distinct(n)
        data = []
        for sample_index in distinct_sample_indices:
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

    def genetic_algorithm(self, mutation_rate=0.1):
        output_directory = "runs"
        run_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        run_path = os.path.join(output_directory, run_id)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        print(f"Run ID: {run_id} has been created at {run_path}")
        initial_prompts = self.generate_init_prompts()
        population = initial_prompts
        bar = tqdm(range(self.generations))
        for gen_id in bar:
            print("Complete Population:",population)
            fitness_scores, questions_list, correct_answers_list, prompt_answers_list = self.evaluate_fitness(population)
            top_prompts, top_prompts_answers_list = self.select_top_prompts(fitness_scores, population, prompt_answers_list)
            df = pd.DataFrame({
                'Prompt': population,
                'Fitness Score': fitness_scores
            })
            df.to_csv(os.path.join(run_path, f'epoch_{gen_id}.csv'), index=False)
            print()
            print("Top Population:", top_prompts)
            print("\n\n")
            new_prompts = self.crossover_using_gpt(top_prompts, questions_list, correct_answers_list, top_prompts_answers_list)
            num_random_prompts = int(self.init_and_fitness_sample * 0.25)
            random_prompts = self.generate_init_prompts(num_random_prompts)
            population = top_prompts + new_prompts + random_prompts
            population = self.mutate_prompts(population, mutation_rate)
            bar.set_description(str({"epoch": gen_id+1, "acc": round(float(np.mean(fitness_scores))*100, 1)}))
        bar.close()

        return population

    def evaluate_fitness(self, prompts):
        distinct_sample_indices = self.sample_distinct(self.init_and_fitness_sample)
        just_questions_list = [self.train_questions_list[int(index)] for index in distinct_sample_indices]
        questions_list = "\n\n".join([str(i+1)+'. """'+self.train_questions_list[int(index)]+'"""' for i,index in enumerate(distinct_sample_indices)])
        correct_answers_list = [self.label_dict[self.train_answers_label[int(i)]] for i in distinct_sample_indices]
        acc_list = []
        prompt_latest_answers_list = []
        for prompt in prompts:
            acc = []
            for retry_id in range(self.num_retries):
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": "Questions:\n\n"+questions_list+"\n\nNote: Ensure you respond with "+str(len(distinct_sample_indices))+" labels."}]
                tmp_function_template = function_templates[1]
                tmp_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["enum"] = [v for _,v in self.label_dict.items()]
                tmp_function_template["parameters"]["properties"]["label_array"]["items"]["properties"]["label"]["description"] += str([v for _,v in self.label_dict.items()])
                tmp_function_template["parameters"]["properties"]["label_array"]["minItems"] = len(distinct_sample_indices)
                tmp_function_template["parameters"]["properties"]["label_array"]["maxItems"] = len(distinct_sample_indices)
                labels = send_query2gpt(self.client, messages, tmp_function_template)
                labels = [l['label'] for l in labels['label_array']]
                accuracy = sum(1 if a == b else 0 for a, b in zip(labels, correct_answers_list)) / len(labels)
                acc.append(accuracy)
                prompt_latest_answers_list.append(labels)
            acc_list.append(sum(acc)/len(acc))
        return acc_list, just_questions_list, correct_answers_list, prompt_latest_answers_list

    def select_top_prompts(self, fitness_scores, population, prompt_answers_list, top_fraction=0.5):
        paired_list = list(zip(population, fitness_scores, prompt_answers_list))
        sorted_prompts = sorted(paired_list, key=lambda x: x[1], reverse=True)
        cutoff = int(len(sorted_prompts) * top_fraction)
        return [prompt for prompt, score, answers_list in sorted_prompts[:cutoff]], [answers_list for prompt, score, answers_list in sorted_prompts[:cutoff]]

    def crossover_using_gpt(self, prompts, questions_list, correct_answers_list, top_prompts_answers_list):
        if len(prompts)<2:
            raise Exception("Too few to cross-over.")
        new_prompts = []
        for i in range(0, len(prompts), 2):
            if i + 1 < len(prompts):
                template = prompts[i]
                additive = prompts[i + 1]
                answers_from_the_two_parent_prompts = top_prompts_answers_list[i:i+2]
                if template.lower().strip()==additive.lower().strip():
                    additive = self.gpt_mutate(additive)
                new_prompt = self.gpt_mix_and_match(template, additive, questions_list, correct_answers_list, answers_from_the_two_parent_prompts)
                new_prompts.append(new_prompt)
        return new_prompts
    
    def gpt_mutate(self, prompt):
        tmp_function_template = function_templates[2]
        tmp_function_template["parameters"]["properties"]["mutated_prompt"]["description"] += str(round(random.random(), 3))
        messages = [{"role": "system", "content": "You are a prompt-mutator as part of an over-all genetic algorithm. Mutate the following prompt while not detracting from the core-task but still rephrasing/mutating the prompt.\n\n"+"Note: For this task the over-arching Problem Description is: "+self.problem_description}, {"role": "user", "content": "Modify the following prompt: \"\"\""+prompt+'"""'}]
        mutated_prompt = send_query2gpt(self.client, messages, tmp_function_template, temperature=random.random()/2+0.5)['mutated_prompt']
        return mutated_prompt

    def gpt_mix_and_match(self, template, additive, questions_list, correct_answers_list, answers_from_parent_prompts):
        example = "\n\n".join(['Question: """'+q+'"""\nIdeal Answer: """'+a+'"""\nYour template parent\'s answer: """'+p_0+'"""\nYour additive parent\'s answer: """'+p_1 for q,a,p_0,p_1 in zip(questions_list[:5], correct_answers_list[:5], answers_from_parent_prompts[0], answers_from_parent_prompts[1])])
        messages = [{"role": "system", "content": "You are a cross-over system as part of an over-all genetic algorithm. You are to ingrain segments of an additive prompt to that of a template/control prompt to create a healthier offspring.\n\n"+"Note: For this task the over-arching Problem Description is: "+self.problem_description+"\n\nExample & History for context:"+example+"\n\nNote: You can use previous mistakes as stepping stones, to quote words/semantics/phrases/keywords/verbs which you think led to the mistake by the AI."}, {"role": "user", "content": "Template Prompt: \"\"\""+template+'"""\n'+'"""Additive Prompt: """'+additive}]
        child_prompt = send_query2gpt(self.client, messages, function_templates[3])['child_prompt']
        return child_prompt

    def mutate_prompts(self, prompts, mutation_rate=0.1):
        mutated_prompts = []
        for prompt in prompts:
            if random.random() < mutation_rate:
                mutated_prompts.append(self.gpt_mutate(prompt))
            else:
                mutated_prompts.append(prompt)
        return mutated_prompts
# GeneticPromptLab
GeneticPromptLab uses genetic algorithms for prompt engineering, enhancing quality and diversity through iterative selection, crossover, and mutation, while efficiently exploring minimal yet diverse samples from the training set.

# README.md for GeneticPromptLab

## Overview

GeneticPromptLab is a Python library designed to harness the power of genetic algorithms for prompt engineering in NLP tasks. By iteratively applying selection, crossover, and mutation processes, GeneticPromptLab enhances the quality and diversity of prompts, leading to improved performance in automated question-answering and classification tasks.

This library specifically leverages Sentence Transformers for embedding generation and k-means clustering to sample minimal yet diverse data from training sets. This strategic sampling ensures efficient exploration and optimization of prompts over multiple generations.

## Features

- **Genetic Algorithm Implementation**: Complete genetic algorithm cycle including initialization, fitness evaluation, selection, crossover, and mutation for prompt engineering.
- **Integration with Sentence Transformers**: Utilizes Sentence Transformers for generating embeddings, enabling effective clustering and sampling.
- **Diverse Sampling**: Uses k-means clustering to select a representative subset of data, ensuring diverse genetic material for algorithm initialization and evolution.
- **Dynamic Mutation and Crossover**: Incorporates custom mutation and crossover operations tailored to prompt characteristics and task requirements.
- **Multi-Dataset Support**: Pre-configured to run experiments on the TREC and AG News datasets, demonstrating its adaptability to different types of text classification tasks.

## Installation

To install GeneticPromptLab, clone this repository and install the required packages:

```bash
git clone https://github.com/username/GeneticPromptLab.git
cd GeneticPromptLab
pip install -r requirements.txt
```

## Quick Start

To run GeneticPromptLab on the AG News dataset:

```python
from GeneticPromptLab import QuestionsAnswersOptimizer, agnews
from openai import OpenAI

# Load API key
with open("openai_api.key", "r") as f:
    key = f.read().strip()

# Initialize client
client = OpenAI(api_key=key)

# Get AG News dataset & Initialize the optimizer
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

# Run the genetic algorithm
optimized_prompts = lab.genetic_algorithm()
print(optimized_prompts)
```

## Benchmarks

### Performance on TREC Dataset

The TREC dataset comprises 4,500 English questions categorized into various labels. Below is the performance improvement graph showing the evolution of prompt effectiveness over 10 generations.

![Performance on TREC Dataset](/images/TREC.png)

### Performance on AG News Dataset

The AG News dataset contains over 1 million news articles classified into categories like World, Sports, Business, and Sci/Tech. Below is the performance graph for the AG News dataset.

![Performance on AG News Dataset](/images/Agnews.png)

## Documentation

- **create_prompts(data)**: Generates prompts based on input data.
- **generate_init_prompts(n)**: Produces initial prompts for the genetic algorithm.
- **sample_distinct(n)**: Samples a distinct subset of data or embeddings.
- **genetic_algorithm(mutation_rate)**: Executes the genetic algorithm process.
- **evaluate_fitness(prompts)**: Evaluates the fitness of each prompt.
- **select_top_prompts(fitness_scores, population, top_fraction)**: Selects the top-performing prompts based on fitness scores.
- **crossover_using_gpt(prompts)**: Performs crossover between prompts.
- **mutate_prompts(prompts, mutation_rate)**: Applies mutations to the given prompts.

## Contributions and Feedback

Contributions, suggestions, and feedback are welcome! If you have any ideas to enhance the app or encounter any issues, please feel free to open an issue or submit a pull request on the GitHub repository. Thank you for your interest in our research work.

## Other Usage Details

### Attribution

* If you use or share this work, please provide attribution with the following information:

_"GeneticPromptLab" by Aman Priyanshu and Supriti Vijay, licensed under the MIT License

* When sharing adaptations of this work, please include a statement indicating that changes were made, such as:

_This work is adapted from "GeneticPromptLab" by Aman Priyanshu and Supriti Vijay, licensed under the MIT License. Original work available at: https://github.com/AmanPriyanshu/GeneticPromptLab_

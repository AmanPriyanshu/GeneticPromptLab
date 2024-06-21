from abc import ABC, abstractmethod

class GeneticPromptLab(ABC):
    @abstractmethod
    def create_prompts(self, data):
        """
        This method should take data and return a list of generated prompts based on the input data.
        """
        pass

    @abstractmethod
    def generate_init_prompts(self, n=None):
        """
        Generate initial prompts for the genetic algorithm, possibly using a subset of the data.
        """
        pass

    @abstractmethod
    def sample_distinct(self, n):
        """
        Sample a distinct subset of the data or embeddings to help in various genetic algorithm processes.
        """
        pass

    @abstractmethod
    def genetic_algorithm(self, mutation_rate=0.1):
        """
        Perform the entire genetic algorithm, including initialization, fitness evaluation, selection, crossover, and mutation.
        """
        pass

    @abstractmethod
    def evaluate_fitness(self, prompts):
        """
        Evaluate the fitness of each prompt in the given list of prompts, returning a list of fitness scores.
        """
        pass

    @abstractmethod
    def select_top_prompts(self, fitness_scores, population, prompt_answers_list, top_fraction=0.5):
        """
        Select the top performing prompts based on fitness scores.
        """
        pass

    @abstractmethod
    def crossover_using_gpt(self, prompts, questions_list, correct_answers_list, top_prompts_answers_list):
        """
        Combine aspects of different prompts to generate new prompts.
        """
        pass

    @abstractmethod
    def mutate_prompts(self, prompts, mutation_rate=0.1):
        """
        Mutate given prompts based on a specified mutation rate.
        """
        pass
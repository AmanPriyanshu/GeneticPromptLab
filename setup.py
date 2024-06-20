from setuptools import setup, find_packages

setup(
    name='genetic-prompt-lab',
    version='1.0.0',
    packages=find_packages(),
    description='GeneticPromptLab uses genetic algorithms for prompt engineering, enhancing quality and diversity through iterative selection, crossover, and mutation, while efficiently exploring minimal yet diverse samples from the training set.',
    long_description=open('README.md').read(),
    url='http://github.com/AmanPriyanshu/GeneticPromptLab',
    author='Aman Priyanshu and Supriti Vijay',
    author_email='amanpriyanshusms2001@gmail.com',
    license='MIT',
    install_requires=[
        # Any dependencies, e.g., 'numpy >= 1.11.0', 'matplotlib'
    ],
    keywords='prompt-engineering genetic-algorithms automated-prompt-generation evolutionary-computation prompt-mutation prompt-crossover AI-optimization adaptive-prompting machine-learning-tools LLM-enhancements',
)

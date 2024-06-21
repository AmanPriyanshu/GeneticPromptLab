from setuptools import setup, find_packages

setup(
    name='genetic-prompt-lab',
    version='1.0.0',
    packages=find_packages(),
    description='GeneticPromptLab uses genetic algorithms for automated prompt engineering (for LLMs), enhancing quality and diversity through iterative selection, crossover, and mutation, while efficiently exploring minimal yet diverse samples from the training set.',
    long_description=open('README.md').read(),
    url='http://github.com/AmanPriyanshu/GeneticPromptLab',
    author='Aman Priyanshu and Supriti Vijay',
    author_email='amanpriyanshusms2001@gmail.com',
    license='MIT',
    install_requires=[
        'pandas==2.2.2',
        'scikit-learn==1.5.0',
        'scipy==1.13.1',
        'sentence-transformers==3.0.1',
        'setuptools==70.1.0',
        'numpy==2.0.0',
        'openai==1.34.0',
        'tqdm==4.66.4',
        'datasets==2.20.0',
        'matplotlib==3.9.0'
    ],
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='llm llms prompt-optimization prompt-engineering genetic-algorithms automated-prompt-generation evolutionary-algorithms evolutionary-algorithm prompt-mutation prompt-crossover AI-optimization prompt prompts prompt-tuning prompt-engineering-tools LLM-enhancements',
)

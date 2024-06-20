function_templates = [
    {
        "name": "generate_prompts",
        "description": "You are a Prompt-Engineering assistant whose objective is to understand the given task and come up with a prompt which enables the model to answer these questions accurately.\n\nNote: Questions will change but the prompt you give will be added as a pre-fix to all questions.\n\nUnderstand that the prompt should be generalized to the problem description and can be generalized across all classes. Ensure you do not entirely regurgitate the prompt but be more specific in cases where you feel an LLM might be confused/incorrect as simple prompts don't give the best results.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "This is a prompt which if given to you, an LLM, it'll be able to correctly classify the given question for the given labels."
                }
            },
            "required": ["prompt"]
        }
    }
]
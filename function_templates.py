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
    },
    {
        "name": "QnA_bot",
        "description": "You are a Question-Answering bot, that chooses the correct label for classification given a question.\nNote: Try to select from the given options.",
        "parameters": {
            "type": "object",
            "properties": {
                "label_array": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "description": "Most correct label based on the given queries and instructions in the same order provided. Labels allowed: ",
                                "type": "string",
                                "enum": []
                            },
                        },
                        "required": ["label"],
                    },
                    "description": "An array of labels.",
                    "minItems": 0,
                    "maxItems": 0,
                }
            },
            "required": ["label_array"]
        }
    },
    {
        "name": "prompt_mutate",
        "description": "You are part of Genetic-Optimization Algorithm whose objective is mutate a prompt to ensure randomess within the given prompt yet it should be a direct derivative of the original. Observe the problem description and make modifications to the original prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "mutated_prompt": {
                    "type": "string",
                    "description": "Re-Edit the prompt based on the following parameter, which essentially discusses the percent of string to be modified between [0.0, 1.0]. Degree to modify: "
                },
            },
            "required": ["mutated_prompt"]
        }
    },
    {
        "name": "prompt_crossover",
        "description": "You are part of Genetic-Optimization Algorithm whose objective is to create a child based on two prompts, specifically a Control Prompt and an Additive Prompt. Such that, a new prompt is created taking the Control Prompt as template to instert segments/important vehicles of the Additive Prompt. Ensure it still satisfies the problem description. Also, ensure the prompt is more detailed, it definitely underperformed which requires us to optimize further, there should be stronger directions and more complex instructions.",
        "parameters": {
            "type": "object",
            "properties": {
                "child_prompt": {
                    "type": "string",
                    "description": "Re-Edit the template/control prompt to create a child prompt which is inspired by the additive prompt. It cannot be the same."
                },
            },
            "required": ["child_prompt"]
        }
    }
]
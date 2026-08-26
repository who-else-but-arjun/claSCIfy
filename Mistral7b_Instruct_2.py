import os
import requests
import re
from typing import List, Optional
import logging
from time import sleep
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIError(Exception):
    pass

# api-inference.huggingface.co (the old text-generation endpoint) has been decommissioned in
# favor of HF's Inference Providers router, which speaks the OpenAI chat-completions format.
# mistralai/Mistral-7B-Instruct-v0.3 has no working provider anymore.
# Using Llama-3.1-8B-Instruct since it's confirmed already enabled and working on this account.
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Validate that all required input fields have the correct types
def validate_inputs(abstract: str, conclusion: str, keywords: List[str], conference_name: str) -> bool:
    if not isinstance(abstract, str):
        raise ValueError("Abstract must be a non-empty string")
    if not isinstance(conclusion, str):
        raise ValueError("Conclusion must be a non-empty string")
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise ValueError("Keywords must be a non-empty list of strings")
    if not isinstance(conference_name, str):
        raise ValueError("Conference name must be a non-empty string")
    return True

# Normalize whitespace and remove special tokens and formatting artifacts
def clean_generated_text(text: str) -> str:
    text = ' '.join(text.split())
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\[.*?\]:', '', text)
    return text.strip()

# Count words by splitting on whitespace
def get_word_count(text: str) -> int:
    return len(text.split())

# Send a system/user prompt pair to the Mistral-7B Instruct API and return the cleaned generated text
def call_mistral_api(system_prompt: str, user_prompt: str, max_length: int = 200, temperature: float = 0.6, max_retries: int = 3, retry_delay: int = 2) -> str:
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_length,
    }

    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        raise APIError("Set the HF_API_TOKEN environment variable with your Hugging Face API token")
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Retry loop with backoff for transient API failures
    for attempt in range(max_retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            generated_text = response_data["choices"][0]["message"]["content"]

            if not generated_text:
                raise APIError("Empty response from API")

            return clean_generated_text(generated_text)

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise APIError(f"Failed to communicate with API after {max_retries} attempts: {str(e)}")
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            sleep(retry_delay)
        
        except (KeyError, IndexError) as e:
            raise APIError(f"Unexpected API response format: {str(e)}")

# Generate a detailed ~100 word justification using abstract, conclusion, and keywords
def generate_initial_justification(
    abstract: str,
    conclusion: str,
    keywords: List[str],
    conference_name: str,
) -> str:
    system_prompt = (
        "You are a precise AI that generates detailed justifications. "
        "Generate a justification of approximately 100 words explaining "
        "why a research paper fits a specific conference. Include specific details "
        "from the abstract and conclusion."
    )
    
    user_prompt = (
        f"Abstract: {abstract}\n\n"
        f"Conclusion: {conclusion}\n\n"
        f"Keywords: {', '.join(keywords)}\n\n"
        f"Conference: {conference_name}\n\n"
        "Generate a detailed justification of around 100 words."
    )

    return call_mistral_api(system_prompt, user_prompt)

# Condense the initial justification into a concise 50-70 word summary
def generate_final_justification(initial_justification: str) -> str:
    system_prompt = (
        "You are a precise AI that creates concise summaries. "
        "Summarize the following justification in EXACTLY 50-70 words while "
        "maintaining the key points and specific details."
    )
    
    user_prompt = (
        f"Original justification:\n{initial_justification}\n\n"
        "Create a concise version between 50-70 words."
    )

    return call_mistral_api(system_prompt, user_prompt, max_length=150)

# Two-stage justification pipeline: generate detailed justification then summarize it
def Doraemon_justification(
    abstract: str,
    conclusion: str,
    keywords: List[str],
    conference_name: str,
) -> Optional[str]:
    try:
        validate_inputs(abstract, conclusion, keywords, conference_name)
        
        # Step 1: Generate initial detailed justification
        logger.info("Generating initial detailed justification...")
        initial_justification = generate_initial_justification(
            abstract, conclusion, keywords, conference_name
        )
        initial_word_count = get_word_count(initial_justification)
        logger.info(f"Initial justification generated: {initial_word_count} words")
        
        # Step 2: Generate final concise justification
        logger.info("Generating final concise justification...")
        final_justification = generate_final_justification(initial_justification)
        final_word_count = get_word_count(final_justification)
        logger.info(f"Final justification generated: {final_word_count} words")
        
        # Verify final length
        if final_word_count < 50 or final_word_count > 70:
            logger.warning(f"Final justification length ({final_word_count} words) outside target range")
        
        return final_justification

    except Exception as e:
        logger.error(f"Error generating justification: {str(e)}")
        raise

def main(abstract, conclusion, keywords, conference_name):
    try:
        justification = Doraemon_justification(abstract, conclusion, keywords, conference_name)
        
        if justification:
            print("\n[INFO] Generated Justification:")
            print("-" * 80)
            print(justification)
            print("-" * 80)
            print(f"Word count: {get_word_count(justification)} words")
        return justification
    except (ValueError, APIError) as e:
        logger.error(f"Failed to generate justification: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    abstract = "This paper presents a novel approach to optimizing neural network architectures using evolutionary algorithms."
    conclusion = "The results demonstrate significant improvements in accuracy and efficiency, making this method suitable for deployment in real-world AI systems."
    keywords = ["neural networks", "evolutionary algorithms", "optimization", "AI"]
    conference_name = "NeurIPS 2025"
    main(abstract, conclusion, keywords, conference_name)
"""
Create OpenAI Batch API requests for generating 2000 stories
This will create a JSONL file with all requests and submit it to OpenAI Batch API
Using gpt-4o-mini:
  - Cost: 50% less than standard API (approximately $0.30 for 2000 stories with batch API)
  - Standard pricing: $0.150 per 1M input tokens, $0.600 per 1M output tokens
  - Batch pricing: $0.075 per 1M input tokens, $0.300 per 1M output tokens
  - Turnaround: Up to 24 hours
  - Limits: 200K TPM, 500 RPM, 10K RPD, 2M TPD
"""
import random
import csv
import json
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(dotenv_path='../../.env')

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
CSV_NAME = 'bigtom/bigtom'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model name')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
parser.add_argument('--max_tokens', type=int, default=450, help='max tokens')
parser.add_argument('--num_shots', type=int, default=3, help='number of shots')
parser.add_argument('--num_stories', type=int, default=2000, help='number of stories to generate')
parser.add_argument('--batch_file', type=str, default='../../data/batch_requests.jsonl', help='output batch file')
args = parser.parse_args()


def create_batch_requests():
    """Create JSONL file with all batch requests"""

    response_template = """Here is the story:
Story: {story}
Aware of event: {awarenes}
Not aware of event: {not_aware}
Action given new state: {action_new}
Action given initial state: {action_init}
Belief Question: {belief_question}
Desire Question: {desire_question}
Action Question: {action_question}
Belief Aware: {belief_answer_aware}
Desire Aware: {desire_answer_aware}
Action Aware: {action_answer_aware}
Belief not Aware: {belief_answer_not_aware}
Desire not Aware: {desire_answer_not_aware}
Action not Aware: {action_answer_not_aware}
Random Event: {random_event}
Aware of random event: {aware_of_random_event}
Not aware of random event: {not_aware_of_random_event}"""

    # Load instruction text
    with open(f'{PROMPT_DIR}/bigtom.txt', 'r') as f:
        instruction_text = f.read()

    # Load examples from existing CSV
    examples = []
    template_var = ["story", "awarenes", "not_aware", "action_new", "action_init", "belief_question", "desire_question", "action_question",
                    "belief_answer_aware", "desire_answer_aware", "action_answer_aware", "belief_answer_not_aware", "desire_answer_not_aware",
                    "action_answer_not_aware", "random_event", "aware_of_random_event", "not_aware_of_random_event"]

    csv_file = f'{DATA_DIR}/{CSV_NAME}.csv'

    with open(csv_file, 'r') as f:
        for line in f.readlines():
            params = line.split(';')
            if len(params) >= len(template_var):
                example = {k: params[v].strip() for v, k in enumerate(template_var)}
                examples.append(example)

    random.shuffle(examples)
    print(f"Loaded {len(examples)} examples from {csv_file}")

    # Create batch requests
    batch_requests = []

    for i in range(args.num_stories):
        letter = random.choice(letters)

        # Create messages with few-shot examples
        messages = [
            {"role": "system", "content": instruction_text}
        ]

        # Add few-shot examples
        for j in range(min(args.num_shots, len(examples))):
            example_idx = (i + j) % len(examples)  # Rotate through examples
            messages.append({"role": "user", "content": "Generate a story"})
            messages.append({"role": "assistant", "content": response_template.format(**examples[example_idx])})

        # Add the actual request
        messages.append({
            "role": "user",
            "content": f'Generate another story, using a different context, object states, and names than the examples did. The name must start with {letter}.'
        })

        # Create batch request in the required format
        batch_request = {
            "custom_id": f"story-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens
            }
        }

        batch_requests.append(batch_request)

    # Write to JSONL file
    with open(args.batch_file, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')

    print(f"Created {len(batch_requests)} batch requests in {args.batch_file}")
    return args.batch_file


def submit_batch(batch_file):
    """Upload the batch file and create a batch job"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Upload the batch file
    print(f"Uploading batch file: {batch_file}")
    with open(batch_file, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    print(f"File uploaded with ID: {batch_input_file.id}")

    # Create the batch job
    print("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Generate 2000 Theory of Mind stories for procedural evaluations"
        }
    )

    print(f"\n{'='*60}")
    print(f"Batch job created successfully!")
    print(f"{'='*60}")
    print(f"Batch ID: {batch_job.id}")
    print(f"Status: {batch_job.status}")
    print(f"Model: {args.model}")
    print(f"Input file ID: {batch_job.input_file_id}")
    print(f"Completion window: {batch_job.completion_window}")
    print(f"\nEstimated completion: Within 24 hours")
    print(f"Estimated cost: ~$0.30 (50% less than standard API)")
    print(f"\nTo check status, run:")
    print(f"  python bigtom_batch_retrieve.py --batch_id {batch_job.id}")
    print(f"  or: ./batch_check_status.sh")
    print(f"{'='*60}\n")

    # Save batch ID to file for later retrieval
    with open('../../data/batch_job_id.txt', 'w') as f:
        f.write(batch_job.id)

    return batch_job


if __name__ == "__main__":
    print(f"Creating batch requests for {args.num_stories} stories...")
    batch_file = create_batch_requests()

    print(f"\nSubmitting batch job to OpenAI...")
    batch_job = submit_batch(batch_file)

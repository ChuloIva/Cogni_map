"""
Async version of bigtom.py with concurrent workers and rate limiting
Generates stories using multiple concurrent API calls while respecting rate limits
"""
import random
import csv
import argparse
import os
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken

from utils import push_data, get_num_items, get_vars_from_out

# Load environment variables from .env file
load_dotenv(dotenv_path='../../.env')

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z']
DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
REPO_URL = 'https://github.com/cicl-stanford/marple_text'
CSV_NAME = 'bigtom/bigtom'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4o', help='model name')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
parser.add_argument('--max_tokens', type=int, default=450, help='max tokens')
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')
parser.add_argument('--num_shots', type=int, default=3, help='number of shots')
parser.add_argument('--num_stories', type=int, default=1000, help='number of stories to generate')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--rpm_limit', type=int, default=450, help='requests per minute limit')
parser.add_argument('--tpm_limit', type=int, default=27000, help='tokens per minute limit')
parser.add_argument('--max_retries', type=int, default=5, help='max retries for rate limit errors')
parser.add_argument('--max_concurrent', type=int, default=10, help='max concurrent requests')

class AsyncRateLimiter:
    """Async rate limiter for OpenAI API calls"""
    def __init__(self, rpm_limit=450, tpm_limit=27000):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_times = []
        self.token_usage = []
        self.lock = asyncio.Lock()

    async def wait_for_availability(self, estimated_tokens=500):
        """Wait until we have capacity for a new request"""
        async with self.lock:
            current_time = time.time()
            cutoff_time = current_time - 60

            # Remove old entries
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]

            # Check RPM limit
            while len(self.request_times) >= self.rpm_limit:
                wait_time = 60 - (current_time - self.request_times[0]) + 1
                print(f"[Rate Limiter] RPM limit reached. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                current_time = time.time()
                cutoff_time = current_time - 60
                self.request_times = [t for t in self.request_times if t > cutoff_time]

            # Check TPM limit
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            while current_tokens + estimated_tokens > self.tpm_limit:
                wait_time = 60 - (current_time - self.token_usage[0][0]) + 1
                print(f"[Rate Limiter] TPM limit reached ({current_tokens}/{self.tpm_limit}). Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                current_time = time.time()
                cutoff_time = current_time - 60
                self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
                current_tokens = sum(tokens for _, tokens in self.token_usage)

            # Record this request
            self.request_times.append(current_time)

    async def record_tokens(self, tokens):
        """Record tokens used in a request"""
        async with self.lock:
            self.token_usage.append((time.time(), tokens))

def estimate_tokens(messages, model="gpt-4o"):
    """Estimate token count for messages"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    num_tokens = 0

    for message in messages:
        num_tokens += tokens_per_message
        if isinstance(message, dict):
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
        else:
            num_tokens += len(encoding.encode(str(message)))

    num_tokens += 3  # Every reply is primed with assistant
    return num_tokens

async def generate_single_story(
    client,
    messages,
    args,
    rate_limiter,
    semaphore,
    story_num
):
    """Generate a single story with retry logic"""
    async with semaphore:
        for attempt in range(args.max_retries):
            try:
                # Estimate tokens for this request
                estimated_tokens = estimate_tokens(messages, args.model) + args.max_tokens

                # Wait for rate limit availability
                await rate_limiter.wait_for_availability(estimated_tokens)

                # Make the API call
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    n=args.num_completions
                )

                # Record token usage
                total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
                await rate_limiter.record_tokens(total_tokens)

                return {
                    'story_num': story_num,
                    'response': response,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }

            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = 2 ** attempt
                    print(f"[Story {story_num}] Rate limit error (attempt {attempt + 1}/{args.max_retries}). Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    if attempt == args.max_retries - 1:
                        raise Exception(f"Max retries exceeded for story {story_num}")
                else:
                    print(f"[Story {story_num}] Error: {error_str}")
                    raise e

    raise Exception(f"Failed to generate story {story_num} after {args.max_retries} attempts")

def prepare_messages(system_message, examples, args, letter):
    """Prepare messages for API call"""
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

    messages = [{"role": "system", "content": system_message}]

    # Add few-shot examples
    for i in range(min(args.num_shots, len(examples))):
        messages.append({"role": "user", "content": "Generate a story"})
        messages.append({"role": "assistant", "content": response_template.format(**examples[i])})

    # Add final prompt
    messages.append({
        "role": "user",
        "content": f'Generate another story, using a different context, object states, and names than the examples did. The name must start with {letter}.'
    })

    return messages

def save_story(story_text, csv_file):
    """Parse and save a generated story to CSV"""
    list_var = ["Story", "Aware of event", "Not aware of event", "Action given new state",
                "Action given initial state", "Belief Question", "Desire Question", "Action Question",
                "Belief Aware", "Desire Aware", "Action Aware", "Belief not Aware",
                "Desire not Aware", "Action not Aware", "Random Event", "Aware of random event",
                "Not aware of random event"]

    out_vars = get_vars_from_out(story_text, list_var)
    data = [out_vars[k] for k in list_var]
    data += ["auto", 0]

    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(data)

async def gen_chat_async(args):
    """Main async function to generate stories with concurrent workers"""
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Initialize rate limiter and semaphore
    rate_limiter = AsyncRateLimiter(rpm_limit=args.rpm_limit, tpm_limit=args.tpm_limit)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Load system prompt
    with open(f'{PROMPT_DIR}/bigtom.txt', 'r') as f:
        instruction_text = f.read()

    # Load examples
    template_var = ["story", "awarenes", "not_aware", "action_new", "action_init",
                   "belief_question", "desire_question", "action_question",
                   "belief_answer_aware", "desire_answer_aware", "action_answer_aware",
                   "belief_answer_not_aware", "desire_answer_not_aware",
                   "action_answer_not_aware", "random_event", "aware_of_random_event",
                   "not_aware_of_random_event"]

    csv_file = f'{DATA_DIR}/{CSV_NAME}.csv'

    examples = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            for line in f.readlines():
                params = line.split(';')
                if len(params) >= len(template_var):
                    example = {k: params[v].strip() for v, k in enumerate(template_var)}
                    examples.append(example)

    if len(examples) < args.num_shots:
        raise Exception(f"Not enough examples in {csv_file}. Need at least {args.num_shots} examples.")

    random.shuffle(examples)

    print(f"\n{'='*60}")
    print(f"Starting async story generation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Stories to generate: {args.num_stories}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Rate limits: {args.rpm_limit} RPM, {args.tpm_limit} TPM")
    print(f"{'='*60}\n")

    # Prepare all tasks
    tasks = []
    for story_num in range(args.num_stories):
        letter = random.choice(letters)
        messages = prepare_messages(instruction_text, examples, args, letter)
        task = generate_single_story(
            client, messages, args, rate_limiter, semaphore, story_num
        )
        tasks.append(task)

    # Track progress
    start_time = time.time()
    completed_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Process tasks as they complete
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            completed_count += 1

            # Extract and save story
            story_text = result['response'].choices[0].message.content
            save_story(story_text, csv_file)

            # Update token counts
            total_prompt_tokens += result['prompt_tokens']
            total_completion_tokens += result['completion_tokens']

            # Calculate price (gpt-4o: $2.50 per 1M input, $10.00 per 1M output)
            price = (total_prompt_tokens * 2.50 + total_completion_tokens * 10.00) / 1000000.

            # Calculate progress
            elapsed = time.time() - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining = args.num_stories - completed_count
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60

            # Print progress
            if completed_count % 10 == 0 or completed_count == args.num_stories:
                print(f"[Progress] {completed_count}/{args.num_stories} stories | "
                      f"Rate: {rate:.1f} stories/s | "
                      f"ETA: {eta_minutes:.1f}m | "
                      f"Cost: ${price:.4f}")

            if args.verbose and completed_count % 50 == 0:
                print(f"\n[Story {result['story_num']}]")
                print(story_text[:200] + "...")
                print()

        except Exception as e:
            print(f"[Error] Failed to generate story: {e}")
            continue

    # Final summary
    elapsed = time.time() - start_time
    total_price = (total_prompt_tokens * 2.50 + total_completion_tokens * 10.00) / 1000000.

    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total stories generated: {completed_count}/{args.num_stories}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {completed_count/elapsed:.2f} stories/second")
    print(f"Total cost: ${total_price:.4f}")
    print(f"Cost per story: ${total_price/completed_count:.6f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Generating {args.num_stories} stories with {args.max_concurrent} concurrent workers")
    if args.verbose:
        print(args)

    asyncio.run(gen_chat_async(args))
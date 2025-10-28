"""
Retrieve results from OpenAI Batch API and process them into the bigtom.csv format
"""
import json
import csv
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(dotenv_path='../../.env')

DATA_DIR = '../../data'
CSV_NAME = 'bigtom/bigtom'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_id', type=str, help='Batch job ID to retrieve')
parser.add_argument('--output_file', type=str, default='../../data/batch_output.jsonl', help='Downloaded batch output file')
parser.add_argument('--check_status', action='store_true', help='Only check status, do not download')
args = parser.parse_args()


def get_vars_from_out(text, list_var):
    """Extract variables from generated text"""
    out_vars = {}
    for var in list_var:
        try:
            start = text.find(var + ':')
            if start == -1:
                out_vars[var] = ""
                continue
            start += len(var) + 1
            end = text.find('\n', start)
            if end == -1:
                out_vars[var] = text[start:].strip()
            else:
                out_vars[var] = text[start:end].strip()
        except:
            out_vars[var] = ""
    return out_vars


def check_batch_status(batch_id):
    """Check the status of a batch job"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    batch_job = client.batches.retrieve(batch_id)

    print(f"\n{'='*60}")
    print(f"Batch Job Status")
    print(f"{'='*60}")
    print(f"Batch ID: {batch_job.id}")
    print(f"Status: {batch_job.status}")
    print(f"Created at: {batch_job.created_at}")
    print(f"Completion window: {batch_job.completion_window}")
    if batch_job.completed_at:
        print(f"Completed at: {batch_job.completed_at}")
    if batch_job.failed_at:
        print(f"Failed at: {batch_job.failed_at}")

    print(f"\nRequest counts:")
    print(f"  Total: {batch_job.request_counts.total}")
    print(f"  Completed: {batch_job.request_counts.completed}")
    print(f"  Failed: {batch_job.request_counts.failed}")

    if batch_job.output_file_id:
        print(f"\nOutput file ID: {batch_job.output_file_id}")

    if batch_job.error_file_id:
        print(f"Error file ID: {batch_job.error_file_id}")

    print(f"{'='*60}\n")

    return batch_job


def download_batch_results(batch_id, output_file):
    """Download batch results from OpenAI"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    batch_job = client.batches.retrieve(batch_id)

    if batch_job.status != "completed":
        print(f"Batch job is not completed yet. Current status: {batch_job.status}")
        return None

    if not batch_job.output_file_id:
        print("No output file available yet.")
        return None

    print(f"Downloading results from output file: {batch_job.output_file_id}")

    # Download the output file
    file_response = client.files.content(batch_job.output_file_id)

    # Save to file
    with open(output_file, 'wb') as f:
        f.write(file_response.content)

    print(f"Results saved to: {output_file}")

    # Also download errors if they exist
    if batch_job.error_file_id:
        print(f"Downloading errors from error file: {batch_job.error_file_id}")
        error_response = client.files.content(batch_job.error_file_id)
        error_file = output_file.replace('.jsonl', '_errors.jsonl')
        with open(error_file, 'wb') as f:
            f.write(error_response.content)
        print(f"Errors saved to: {error_file}")

    return output_file


def process_batch_results(output_file):
    """Process batch results and add to bigtom.csv"""
    list_var = ["Story", "Aware of event", "Not aware of event", "Action given new state", "Action given initial state",
                "Belief Question", "Desire Question", "Action Question",
                "Belief Aware", "Desire Aware", "Action Aware", "Belief not Aware",
                "Desire not Aware", "Action not Aware", "Random Event", "Aware of random event", "Not aware of random event"]

    story_file = f'{DATA_DIR}/{CSV_NAME}.csv'

    success_count = 0
    error_count = 0

    print(f"\nProcessing batch results...")

    with open(output_file, 'r') as f:
        for line in f:
            result = json.loads(line)

            if result['response']['status_code'] != 200:
                error_count += 1
                print(f"Error in {result['custom_id']}: {result['response']}")
                continue

            try:
                # Extract the generated text
                generated_text = result['response']['body']['choices'][0]['message']['content']

                # Parse the generated text
                out_vars = get_vars_from_out(generated_text, list_var)

                # Create data row
                data = [out_vars[k] for k in list_var]
                data += ["auto", 0]

                # Append to CSV
                with open(story_file, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(data)

                success_count += 1

                if success_count % 100 == 0:
                    print(f"Processed {success_count} stories...")

            except Exception as e:
                error_count += 1
                print(f"Error processing {result['custom_id']}: {e}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count} stories")
    print(f"Errors: {error_count}")
    print(f"Stories saved to: {story_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Load batch ID from file if not provided
    if not args.batch_id:
        batch_id_file = '../../data/batch_job_id.txt'
        if os.path.exists(batch_id_file):
            with open(batch_id_file, 'r') as f:
                args.batch_id = f.read().strip()
            print(f"Using batch ID from file: {args.batch_id}")
        else:
            print("Error: No batch ID provided and no saved batch ID found.")
            print("Usage: python bigtom_batch_retrieve.py --batch_id <batch_id>")
            exit(1)

    # Check status
    batch_job = check_batch_status(args.batch_id)

    if args.check_status:
        exit(0)

    # Download and process if completed
    if batch_job.status == "completed":
        output_file = download_batch_results(args.batch_id, args.output_file)
        if output_file:
            process_batch_results(output_file)
    else:
        print(f"Batch is not ready yet. Current status: {batch_job.status}")
        print(f"Run this script again later to check status and download results.")
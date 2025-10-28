 Batch ID: batch_6900dab510888190b3dc8541bdae36f9
  Estimated Cost: ~$0.30
  Estimated Completion: Within 24 hours (usually 2-6 hours for 2000 requests)

  To Check Status:

  Navigate to the scripts directory and run:
  cd /Users/ivanculo/Desktop/Projects/Cogni_map/brije/ToM/procedural-evals-tom/code/scripts
  ./batch_check_status.sh

  Or from anywhere:
  cd /Users/ivanculo/Desktop/Projects/Cogni_map/brije/ToM/procedural-evals-tom/code/src
  uv run bigtom_batch_retrieve.py --check_status

  To Retrieve Results (when status is "completed"):

  cd /Users/ivanculo/Desktop/Projects/Cogni_map/brije/ToM/procedural-evals-tom/code/scripts
  ./batch_retrieve_results.sh

  This will:
  1. Download the batch results
  2. Process them into data/bigtom/bigtom.csv
  3. Generate the 6 condition directories for your 3 tasks (Forward Belief, Forward Action,
  Backward Belief) with true_belief and false_belief conditions

  Notes:

  - The batch ID is saved in data/batch_job_id.txt so you don't need to remember it
  - You can check status as many times as you want
  - Only retrieve when status shows "completed"
  - The scripts will tell you exactly what's happening at each step

  You're all set! Just check back in a few hours and retrieve the results when ready.
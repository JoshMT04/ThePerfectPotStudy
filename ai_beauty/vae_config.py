
NUM_ROUNDS = 5
N_GENERATIONS = 5

PROMPT = "From these two images of vases, please select the one that is most beautiful. Please take time to think very carefully before responding. Return a single integer, either '1' or '2', strictly enclosed within <selection></selection> tags."
FINAL_PROMPT = f"From these {NUM_ROUNDS} images of vases, please select the one that is most beautiful. Please take time to think very carefully before responding. Return a single integer between 1 and {NUM_ROUNDS}, strictly enclosed within <selection></selection> tags."

BUDGET_TOKENS = 5000

DRIVE_FOLDER_ID = "1RbwakbNcEro_L96UTWj6E6K7e8vo0k9a"

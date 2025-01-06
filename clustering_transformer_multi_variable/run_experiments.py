import subprocess

commands = [
    "python3 main.py --seq_len 50 --pred_len 5  --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 10 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 15 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 20 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 25 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 30 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 35 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 40 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 45 --attention_masking 1 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 50 --attention_masking 1 --time_enc 1",

    "python3 main.py --seq_len 50 --pred_len 5  --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 10 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 15 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 20 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 25 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 30 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 35 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 40 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 45 --attention_masking 1 --time_enc 0",
    "python3 main.py --seq_len 50 --pred_len 50 --attention_masking 1 --time_enc 0",

    "python3 main.py --seq_len 50 --pred_len 5  --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 10 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 15 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 20 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 25 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 30 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 35 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 40 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 45 --attention_masking 0 --time_enc 1",
    "python3 main.py --seq_len 50 --pred_len 50 --attention_masking 0 --time_enc 1",

]

for cmd in commands:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
import subprocess

commands = [
    "python3 main.py --seq_len 50 --pred_len 5",
    "python3 main.py --seq_len 50 --pred_len 10",
    "python3 main.py --seq_len 50 --pred_len 15",
    "python3 main.py --seq_len 50 --pred_len 20",
    "python3 main.py --seq_len 50 --pred_len 25",
    "python3 main.py --seq_len 50 --pred_len 30",
    "python3 main.py --seq_len 50 --pred_len 35",
    "python3 main.py --seq_len 50 --pred_len 40",
    "python3 main.py --seq_len 50 --pred_len 45",
    "python3 main.py --seq_len 50 --pred_len 50",
]

for cmd in commands:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
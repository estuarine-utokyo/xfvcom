from pathlib import Path

value_str = "flux_by_node.csv"
print(f"Checking if '{value_str}' exists...")
print(f"Path(value_str).exists() = {Path(value_str).exists()}")
print(f"Current working directory: {Path.cwd()}")
print(f"Files in current directory: {list(Path('.').glob('*.csv'))}")

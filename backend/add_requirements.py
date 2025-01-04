import subprocess

# Path to your requirements.txt file
requirements_file = "requirements.txt"

# Read dependencies from requirements.txt
with open(requirements_file, "r") as file:
    dependencies = file.readlines()

# Install each dependency using poetry add
for dependency in dependencies:
    dependency = dependency.strip()  # Remove whitespace/newlines
    if dependency:  # Skip empty lines
        print(f"Adding {dependency}...")
        subprocess.run(["poetry", "add", dependency], check=True)

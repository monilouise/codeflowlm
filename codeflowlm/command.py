import subprocess

def execute_command(command):
    print(command)
    result = subprocess.run(command, shell=True, text=True)
    print(result.stdout)
    if result.stderr:
      print("Error:", result.stderr)
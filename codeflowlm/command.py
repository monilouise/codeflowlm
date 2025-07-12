import subprocess

"""
def execute_command(command):
    print(command)
    result = subprocess.run(command, shell=True, text=True)
    print(result.stdout)
    if result.stderr:
      print("Error:", result.stderr)
"""
   
def execute_command(command):
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Lê e imprime cada linha assim que disponível
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)  # flush força o print imediato no Colab

    process.stdout.close()
    process.wait()
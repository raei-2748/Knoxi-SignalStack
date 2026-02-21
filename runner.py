import builtins
import warnings

def display(x):
    print(x)

builtins.display = display
warnings.filterwarnings('ignore')

with open('run_nb.py') as f:
    exec(f.read(), globals())

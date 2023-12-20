from federated_learning import run_experiment
from arguments import Arguments

def main():
    args1 = Arguments()
    args1.epochs = 50
    args2 = Arguments()
    args2.epochs = 50
    args2.topology = 'single'
    
    args = [args1, args2]
    for a in args:
        run_experiment(a)

if __name__ == '__main__':
    main()

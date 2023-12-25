from federated_learning import run_experiment
from arguments import Arguments

def main():

    args1 = Arguments()
#    args1.epochs = 50
#    args2 = Arguments()
#    args2.epochs = 1500
#    args2.topology = 'single'
#
#
#    args3 = Arguments()
#    args3.epochs = 1500
#    args3.num_attackers = 2
#    args4 = Arguments()
#    args4.epochs = 1500
#    args4.num_attackers = 2
#    args4.topology = 'single'
    
    #args = [args1, args2, args3, args4]
    args = [args1]
    for a in args:
        run_experiment(a)

if __name__ == '__main__':
    main()

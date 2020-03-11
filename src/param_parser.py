import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Bitcoin OTC dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description="Run SGCN.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training epochs. Default is 100.")
    
    parser.add_argument("--seed",
                    type=int,
                    default=42,
	                help="Random seed for sklearn pre-training. Default is 42.")
    
    parser.add_argument("--no-cuda", 
                    action='store_true', 
                    default=False,
                    help='Disables CUDA training.')
    
    parser.add_argument('--fastmode', 
                    action='store_true', 
                    default=False,
                    help='Validate during training pass.')
    
    
    


    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30,
	                help="Number of SVD iterations. Default is 30.")

    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=64,
	                help="Number of SVD feature extraction dimensions. Default is 64.")



    parser.add_argument("--lamb",
                        type=float,
                        default=1.0,
	                help="Embedding regularization parameter. Default is 1.0.")

    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
	                help="Test dataset size. Default is 0.2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-5,
	                help="Learning rate. Default is 10^-5.")

    parser.add_argument("--spectral-features",
                        dest="spectral_features",
                        action="store_true")

    parser.add_argument("--general-features",
                        dest="spectral_features",
                        action="store_false")

    parser = argparse.ArgumentParser()

    parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')


    parser.set_defaults(spectral_features=True)


    return parser.parse_args()

import argparse

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Physical Reasoning')
    parser.add_argument("--task_name", default="exp1")
    parser.add_argument("--data_path", default="", type=str, help="data path")
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument("--batch_size_train", default=32, type=int, help="Training batch size")
    parser.add_argument("--batch_size_test", default=32, type=int, help="Testing batch size")
    parser.add_argument("--n_epochs", default=20, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.5, type=float)
    parser.add_argument("--optimizer", default="sgd", type=str, help="specify optimizer")
    parser.add_argument("--loss", default="ce", type=str, help="specify loss")

    args = parser.parse_args()
    return args
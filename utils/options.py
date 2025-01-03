import argparse

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Physical Reasoning')
    parser.add_argument("--task_name", default="exp1")
    parser.add_argument('--network_file', type=str, required=True, 
                        help='Name of the network file to import')
    parser.add_argument("--picture_path", default="", type=str, help="picture path")
    parser.add_argument("--metadata_path", default="", type=str, help="metadata path")
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument("--batch_size_train", default=32, type=int, help="Training batch size")
    parser.add_argument("--batch_size_test", default=32, type=int, help="Testing batch size")
    parser.add_argument("--n_epochs", default=20, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.5, type=float)
    parser.add_argument("--warmup", default="False", type=str, help="warmup")
    parser.add_argument("--resume_path", default="", type=str, help="path to resume training")
    parser.add_argument("--n_fold", default=10, type=int, help="number of folds for cross-validation")

    parser.add_argument("--optimizer", default="sgd", type=str, help="specify optimizer")
    parser.add_argument("--loss", default="ce", type=str, help="specify loss")

    args = parser.parse_args()
    return args
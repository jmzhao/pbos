

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="path to dataset")
    parser.add_argument(
        '--embeddings',
        required=True,
        help="path to word embeddings"
    )
    parser.add_argument(
        '--results_dir',
        help="path to the results directory",
        default="results/pos_reg_search"
    )
    args = parser.parse_args()
    main(args.results_dir, args.embeddings, args.dataset)

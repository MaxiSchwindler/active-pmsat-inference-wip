import sys

import pandas as pd
import lightgbm as lgb
import numpy as np
import ast

from active_pmsatlearn.log import get_logger
from evaluation.utils import new_file

logger = get_logger(__name__)


def load_data_from_csv(csv_file):
    logger.info(f'Loading data from {csv_file}')
    df = pd.read_csv(csv_file, comment='#')

    # extract comment which tells us stat names
    with open(csv_file, 'r') as f:
        while not (line := f.readline()).startswith("#"):
            pass
        stat_names = ast.literal_eval(line.split("# stats: ")[1])

    automata_results = []
    rankings = []
    group = []

    for r_idx, row in df.iterrows():
        try:
            def str_to_py(string):
                if string in (None, np.nan):
                    return None
                else:
                    string = string.replace('nan', 'None')
                    return ast.literal_eval(string)
            stats = [str_to_py(row[num_states_str]) for num_states_str in row.keys() if num_states_str != 'ranking']
            ranking = ast.literal_eval(row['ranking'])

            num = len(stats)
            for i in reversed(range(num)):
                if stats[i] is None:
                    stats.pop(i)
                    ranking.pop(i)

        except Exception as e:
            raise type(e)(f"{str(e)} (occurred in row {r_idx}:\n{row}\n)")

        automata_results.extend(stats)
        rankings.extend(ranking)
        group.append(len(stats))

    dataset = lgb.Dataset(np.array(automata_results), label=np.array(rankings), group=group, feature_name=stat_names)
    return dataset


def train_model(csv_file):
    dtrain = load_data_from_csv(csv_file)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "num_leaves": 16,
        "learning_rate": 0.1,
    }
    logger.info(f'Training model')
    ranker = lgb.train(params, dtrain, num_boost_round=100)

    model_file = new_file("lgb_model.txt")
    logger.info(f'Saving model to {model_file}')
    ranker.save_model(model_file)
    return ranker


def load_model(model_file):
    logger.info(f'Loading model from {model_file}')
    ranker = lgb.Booster(model_file=model_file)
    return ranker


def main(*args):
    csv_file = args[0]
    train_model(csv_file)


if __name__ == '__main__':
    main(*sys.argv[1:])
from typing import List

import pandas as pd
import datetime
import os

from Model import TrainableModel
from Utilities import ElapsedTime, mkdir


def findOptimalBatchSize(model: TrainableModel, max_batch_size: int = 512, n_epoch: int = 3, save_excel: bool = True):
    current_batch_size = max_batch_size
    fastest_idx = None
    tested_current_batch_size_arr = []
    runtime_arr = []
    while current_batch_size > 0:
        try:
            runtime = _run_configuration(model, current_batch_size, n_epoch)
            tested_current_batch_size_arr.append(current_batch_size)
            runtime_arr.append(runtime)
            if fastest_idx is None or runtime < runtime_arr[fastest_idx]:
                fastest_idx = len(tested_current_batch_size_arr) - 1
                current_batch_size /= 2
            else:
                break
        except Exception:
            tested_current_batch_size_arr.append(current_batch_size)
            runtime_arr.append(-1)
            current_batch_size /= 2

    if save_excel:
        _save_results_to_excel(model, runtime_arr, tested_current_batch_size_arr)

    return tested_current_batch_size_arr[fastest_idx]


def _run_configuration(model: TrainableModel, current_batch_size: int, n_epoch: int) -> float:
    t = ElapsedTime(process_name=str(current_batch_size), verbose=False)
    with t:
        model.train(batch_size=current_batch_size, n_epoch=n_epoch)
    return t.elapsed_time_ms


def _save_results_to_excel(model: TrainableModel, runtime_arr: List[float], tested_current_batch_size_arr: List[int]):
    df = pd.DataFrame({'BatchSize': tested_current_batch_size_arr, 'Runtime': runtime_arr})
    REPORTS_DIR = 'reports'
    mkdir(REPORTS_DIR)
    file_name = '{}_{}.xlsx'.format(model.name, datetime.datetime.now().isoformat())
    file_name_ful = os.path.join(REPORTS_DIR, file_name)
    df.to_excel(file_name_ful)

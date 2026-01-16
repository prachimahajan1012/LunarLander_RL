#!/usr/bin/env python3
"""
Summary from evaluation data.
Generates file 'evaluation_report.txt' under experiments/learning_curves_analysis/
"""

import numpy as np
from pathlib import Path

experiments_dir = Path('experiments')

dense_eval = np.load(experiments_dir / 'dense' / 'evaluations.npz')
sparse_eval = np.load(experiments_dir / 'sparse' / 'evaluations.npz')

report = []

report.append('\n' + '='*80)
report.append('QUICK STATISTICS FROM EVALUATION DATA')
report.append('='*80)

report.append('\nDENSE REWARD:')
report.append(f'  Timesteps evaluated:  {len(dense_eval["timesteps"])}')
report.append(f'  Training range:       {dense_eval["timesteps"][0]:,} to {dense_eval["timesteps"][-1]:,} steps')
report.append(f'  Episodes per step:    {dense_eval["results"].shape[1]}')
dense_returns = dense_eval['results'].mean(axis=1)
dense_steps = dense_eval['timesteps']
report.append(f'  Initial return:       {dense_returns[0]:>7.1f}')
report.append(f'  Final return:         {dense_returns[-1]:>7.1f}')
report.append(f'  Best return:          {np.max(dense_returns):>7.1f}')

report.append('\nSPARSE REWARD:')
report.append(f'  Timesteps evaluated:  {len(sparse_eval["timesteps"])}')
report.append(f'  Training range:       {sparse_eval["timesteps"][0]:,} to {sparse_eval["timesteps"][-1]:,} steps')
report.append(f'  Episodes per step:    {sparse_eval["results"].shape[1]}')
sparse_returns = sparse_eval['results'].mean(axis=1)
sparse_steps = sparse_eval['timesteps']
report.append(f'  Initial return:       {sparse_returns[0]:>7.1f}')
report.append(f'  Final return:         {sparse_returns[-1]:>7.1f}')
report.append(f'  Best return:          {np.max(sparse_returns):>7.1f}')

output_dir = experiments_dir / 'learning_curves_analysis'
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / 'evaluation_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
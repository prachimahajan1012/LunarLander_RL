# Usage
```
python .\lunarlander_reward_shaping_experiment.py --n_seeds 1 --n_timesteps 20000 --record_videos --video_dir experiments\videos_sparse1
```
\videos_test3- 200 eval_steps with sparse reward +100 -10 and step penalty, dense fuel penalty -0.05

\videos_test4- evaluating by reducing fuel penalty from -0.05 to -0.01 for dense, removing step penalty for sparse and using +100 and -100 for reward, doing 400 eval steps
dense ep 349 thrusters are firing, lander always takes off
sparse- ep 115 thrusters are firing, lander always takes off

\videos_test5- removed fuel and step penalty from dense and sparse changed sparse reward to +200 -100
sparse- ep 29, ep 374 (thrusters are firing, lander is crashing)
dense- thrusters are firing, lander always takes off

\videos_eval_sparse_retrained- step penalty removed, thrusters left, right, bottom fire now, lander lands correctly sometimes, sometimes in random places,
performance does not exactly get better with episodes its random.

\videos_sparse1- only evaluation without training. with rewards +100 and -100, no fuel penalty. Thrusters are firing (bottom only), lander takes off every time. 
\videos_trained_sparse1 - same parameters as above but with the best model. All thrusters are firing, lander lands but only sometimes in the right place. 

\videos_sparse2- only evaluation without training. with rewards +200 and -100, no fuel penalty. Thrusters are firing (sides only), lander crashes every time in random places.
\videos_trained_sparse2 - same parameters as above but with the best model. All thrusters are firing, lander lands only sometimes in the right place, but usually close to the landing pad.

\videos_sparse3- only evaluation without training. with rewards +150 and -100, no fuel penalty. Thrusters are firing (sides only), lander crashes every time in random places.
\videos_trained_sparse3 - same parameters as above but with the best model. All thrusters are firing, lander lands only sometimes in the right place, but usually close to the landing pad.
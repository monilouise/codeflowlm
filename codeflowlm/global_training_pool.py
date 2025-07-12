import os
import pickle


def get_global_training_pool():
  if os.path.exists('global_training_pool.pkl'):
    with open('global_training_pool.pkl', 'rb') as f:
      global_training_pool = pickle.load(f)
  else:
    global_training_pool = []

  return global_training_pool
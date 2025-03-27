import time

def measure_time(sec, name, Tune=False):
    """ Measure the time taken for training the model """
    hours, rem = divmod(sec, 3600)
    minutes, seconds = divmod(rem, 60)
    if Tune:
        print(f"{name} Hyperparameter Tuning Time: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")
    else:
        print(f"{name} Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")


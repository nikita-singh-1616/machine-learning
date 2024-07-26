# you can save a trained model to file in two ways ...using pickel or by using joblib
# Saving the trained model to a file
import pickle
import joblib

def save_using_pickle(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def retrieving_pickle_model(file_name):
    with open(file_name, 'rb') as f:
        loaded_file = pickle.load(f)
    return loaded_file


def saving_using_job_lib(model, file_name):
    joblib.dump(model, file_name)


def retrieving_joblib_model(file_name):
    joblib_model = joblib.load(file_name)
    return joblib_model

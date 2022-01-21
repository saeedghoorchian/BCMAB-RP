from data_loading.r6b_dataset import Dataset
import gc
import pickle

R6B_FILES = (
    [
        "dataset/r6b/ydata-fp-td-clicks-v2_0.20111010",
        "dataset/r6b/ydata-fp-td-clicks-v2_0.20111011",
    ]
)

R6B_PICKLE_FILES = ["dataset/r6b/data_10.pickle"]


def get_r6b_data():
    print(f"Loading R6B data from files\n{R6B_FILES}")

    dataset = Dataset()
    dataset.fill_yahoo_events_second_version_r6b(
        filenames=R6B_FILES,
    )

    print("Loading done.")
    return dataset


def get_r6b_pickle_data():
    print(f"Loading R6B data from files\n{R6B_PICKLE_FILES[0]}")
    with open(R6B_PICKLE_FILES[0], "rb") as f:
        gc.disable()
        dataset = pickle.load(f)
        gc.enable()
    print("Loading done.")
    return dataset
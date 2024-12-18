# trunk-ignore(bandit/B403)
import pickle


def read_pickle_file(file_path):
    """
    Read data from a pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        The unpickled data
    """
    try:
        with open(file_path, "rb") as f:
            # trunk-ignore(bandit/B301)
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading pickle file: {str(e)}")
        return None


if __name__ == "__main__":
    file_path = "daphnet/processed/daphnet_train.pkl"
    # )
    data = read_pickle_file(file_path)
    if data is not None:
        print("Data loaded successfully")

    print(data.shape)
    print(data[:5] * 20)

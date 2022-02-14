from sklearn import preprocessing
from pathlib import Path

pathdata = "datasets/bases"
datasets = {"brodatz": f"{pathdata}/brodatz",
            "uiuc": f"{pathdata}/UIUC",
            # todo: check for outex13 because it has a different structure
            "outex13": f"{pathdata}/Outex_TC_00013/images"}


class DataSource:
    def __init__(self, name):
        self.name = name
        self.files = []
        self.labels = []
        path = Path(datasets[name])
        self.read(path)

    def read(self, path):
        """ Read classes and samples of a dataset with the structure:
        dataset/
            class1/
                sample1
                sample2
                ...
            class2/
            ...
        """
        for d in path.iterdir():
            for f in d.iterdir():
                self.files.append(f)
                self.labels.append(d.name)

        # encode string labels as numbers
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}\n<{self.name!r}>'


if __name__ == "__main__":
    ds_name = "brodatz"
    ds = DataSource(ds_name)
    print(ds.files)
    print(ds.labels)

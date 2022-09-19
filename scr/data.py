from torch.utils.data import DataLoader
from language_models.pretrainedbert.roberta import preprocess
from transformers import AutoTokenizer
from pytorch_lightning import LightningDataModule
from functools import partial
from datasets import ClassLabel, Dataset
import pandas as pd
import torch

class TweetDataSet(LightningDataModule):
    def __init__(
        self, data_path, test_size, val_size, batch_size, tokenizer_path, num_workers
    ):
        super(TweetDataSet, self).__init__()
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
        DATASET_ENCODING = "ISO-8859-1"
        self.data_csv = pd.read_csv(self.data_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
        self.data_csv.target = self.data_csv.target.replace(4, 1)
        self.data_csv['date'] = pd.to_datetime(self.data_csv['date'])
        self.data = Dataset.from_pandas(self.data_csv)
        self.data = self.data.cast_column('target', ClassLabel(num_classes=2, names=[0, 1]))
        self.data = self.data.train_test_split(
            test_size=self.test_size, stratify_by_column="target", seed=1234
        )
        print(self.data['train'].features)


        self.test = self.data["test"]
        self.train = self.data["train"].train_test_split(
            test_size=self.val_size, stratify_by_column="target", seed=1234
        )
        self.val = self.train["test"]
        self.train = self.train["train"]
        print(
            f"train size: {len(self.train)}",
            f"validation size: {len(self.val)}",
            f"test size: {len(self.test)}",
        )

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.collator_fn = partial(collator, tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collator_fn,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collator_fn,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            collate_fn=self.collator_fn,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


def collator(batch, tokenizer):
    targets = torch.LongTensor([ex["target"] for ex in batch]).unsqueeze(1)
    date = torch.LongTensor([[ex["date"].day - 1, ex["date"].month - 1] for ex in batch])
    days = date[:,0]
    months = date[:,1]
    texts = [ex["text"] for ex in batch]
    texts = [preprocess(item) for item in texts]
    out_tokenizer = tokenizer(texts, truncation=True, padding=True, max_length=128)
    input_ids = torch.LongTensor(out_tokenizer['input_ids'])
    attention_mask = torch.LongTensor(out_tokenizer['attention_mask'])
    return input_ids, days, months, attention_mask, targets

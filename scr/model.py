import torch.nn as nn
from pytorch_lightning.core.module import LightningModule
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel
from torchmetrics import Accuracy
import torch

class TweetCatModel(LightningModule):
    def __init__(
        self, hidden_dim, dropout_clf, output_dim, learning_rate, max_epochs, lm_path, day_emb_dim, month_emb_dim
    ):
        super(TweetCatModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_clf = dropout_clf
        self.output_dim = output_dim
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.day_embeddings = nn.Embedding(31, day_emb_dim)
        self.month_embeddings = nn.Embedding(12, month_emb_dim)

        self.lm = AutoModel.from_pretrained(lm_path)
        lm_output = self.lm.pooler.dense.out_features

        self.classifier = nn.Sequential(
            nn.Linear(lm_output + day_emb_dim + month_emb_dim, hidden_dim),
            nn.Dropout(dropout_clf),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Define metrics used during training
        self.metric = Accuracy()
        self.eval_metric = Accuracy()
        self.test_metric = Accuracy()

    def forward(self, input_ids, days, months, attenstion_mask):
        lm_output = self.lm(input_ids, attenstion_mask).pooler_output
        days = self.day_embeddings(days)
        months = self.month_embeddings(months)
        features = torch.cat([lm_output, days, months], dim=1)
        logits = self.classifier(features)
        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels.float())

    def configure_optimizers(self):
        tagger_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = AdamW(tagger_params, lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.max_epochs / 10),
            num_training_steps=self.max_epochs,
        )
        return [optimizer], [scheduler]

    def decode(self, scores):
        # scores (float) [batch_size, output_dim]
        res = torch.argmax(scores, axis=1)
        return res

    def training_step(self, batch, batch_idx):
        inputs_ids, days, months, attenstion_mask, label_ids = batch
        logits = self(inputs_ids, days, months, attenstion_mask)
        loss = self.compute_loss(logits, label_ids)
        preds = logits.unsqueeze(1)
        self.metric(preds, label_ids)
        # Log metrics into logger
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs_ids, days, months, attenstion_mask, label_ids = batch
        logits = self(inputs_ids, days, months, attenstion_mask)
        loss = self.compute_loss(logits, label_ids)
        preds = logits.unsqueeze(1)
        self.eval_metric(preds, label_ids)
        # Log metrics into logger
        self.log("val_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_epoch_end(self, outs):
        self.log("val_acc", self.eval_metric.compute())

    def test_step(self, batch, batch_idx):
        inputs_ids, days, months, attenstion_mask, label_ids = batch
        logits = self(inputs_ids, days, months, attenstion_mask)
        loss = self.compute_loss(logits, label_ids)
        preds = logits.unsqueeze(1)
        self.test_metric(preds, label_ids)
        # Log metrics into logger
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss
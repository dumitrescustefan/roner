import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss, CrossEntropyLoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from nervaluate import Evaluator
import numpy as np


class Model(pl.LightningModule):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", tokenizer_name=None, lr=2e-05,
                 model_max_length=512, bio2tag_list=[], tag_list=[]):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer_name = tokenizer_name
        if tokenizer_name is None or tokenizer_name == "":
            self.tokenizer_name = model_name

        print("Loading AutoModel [{}] ...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, strip_tokens=False)
        self.model_token_classification = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(bio2tag_list))

        self.lr = lr
        self.model_max_length = model_max_length
        self.bio2tag_list = bio2tag_list
        self.tag_list = tag_list
        self.num_labels = len(bio2tag_list)
        self.to_device = None

        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

        self.bio2tag_list = bio2tag_list
        self.tag_list = tag_list

        if self.tokenizer.cls_token_id is None:
            print(
                f"*** Warning, tokenizer {self.tokenizer_name} has no defined CLS token: sequences will not be marked with special chars! ***")
        if self.tokenizer.sep_token_id is None:
            print(
                f"*** Warning, tokenizer {self.tokenizer_name} has no defined SEP token: sequences will not be marked with special chars! ***")
        if self.tokenizer.pad_token_id is None:
            print(
                f"*** Warning, tokenizer {self.tokenizer_name} has no defined PAD token: sequences will be padded with 0 by default! ***")


    def forward(self, input_ids, attention_mask, labels):
        output = self.model_token_classification(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return output["loss"], output["logits"]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, logits = self(input_ids, attention_mask, labels)
        self.train_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_idx = batch["token_idx"]

        loss, logits = self(input_ids, attention_mask, labels)  # logits is [batch_size, seq_len, num_classes]

        batch_size = logits.size()[0]
        batch_pred = torch.argmax(logits.detach().cpu(), dim=-1).tolist()  # reduce to [batch_size, seq_len] as list
        batch_gold = labels.detach().cpu().tolist()  # [batch_size, seq_len] as list
        batch_token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size):
            pred, gold, idx = batch_pred[batch_idx], batch_gold[batch_idx], batch_token_idx[batch_idx]
            y_hat, y = [], []
            for i in range(0, max(idx) + 1): # for each sentence
                pos = idx.index(i)  # find next token index and get pred and gold
                y_hat.append(pred[pos])
                y.append(gold[pos])
            self.valid_y_hat.append(y_hat)
            self.valid_y.append(y)

        self.valid_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        print()
        mean_val_loss = sum(self.valid_loss) / len(self.valid_loss)
        gold, pred = [], []
        for y, y_hat in zip(self.valid_y, self.valid_y_hat):
            gold.append([self.bio2tag_list[token_id] for token_id in y])
            pred.append([self.bio2tag_list[token_id] for token_id in y_hat])

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")

        results, results_by_tag = evaluator.evaluate()
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/ent_type", results["ent_type"]["f1"])
        self.log("valid/partial", results["partial"]["f1"])
        self.log("valid/strict", results["strict"]["f1"])
        self.log("valid/exact", results["exact"]["f1"])

        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_idx = batch["token_idx"]

        loss, logits = self(input_ids, attention_mask, labels)  # logits is [batch_size, seq_len, num_classes]

        batch_size = logits.size()[0]
        batch_pred = torch.argmax(logits.detach().cpu(), dim=-1).tolist()  # reduce to [batch_size, seq_len] as list
        batch_gold = labels.detach().cpu().tolist()  # [batch_size, seq_len] as list
        batch_token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size):
            pred, gold, idx = batch_pred[batch_idx], batch_gold[batch_idx], batch_token_idx[batch_idx]
            y_hat, y = [], []
            for i in range(0, max(idx) + 1):  # for each sentence
                pos = idx.index(i)  # find next token index and get pred and gold
                y_hat.append(pred[pos])
                y.append(gold[pos])
            self.test_y_hat.append(y_hat)
            self.test_y.append(y)

        self.test_loss.append(loss.detach().cpu().numpy())

    def test_epoch_end(self, outputs):
        mean_val_loss = sum(self.test_loss) / len(self.test_loss)
        gold, pred = [], []
        for y, y_hat in zip(self.test_y, self.test_y_hat):
            gold.append([self.bio2tag_list[token_id] for token_id in y])
            pred.append([self.bio2tag_list[token_id] for token_id in y_hat])

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")

        results, results_by_tag = evaluator.evaluate()
        self.log("test/avg_loss", mean_val_loss, prog_bar=True)
        self.log("test/ent_type", results["ent_type"]["f1"])
        self.log("test/partial", results["partial"]["f1"])
        self.log("test/strict", results["strict"]["f1"])
        self.log("test/exact", results["exact"]["f1"])

        import pprint
        print("_" * 120)
        print("\n\n Test results: \n")
        pprint.pprint(results["strict"])
        print("\n Per class Strict-F1 values:")
        for cls in self.tag_list:
            print(f'\t {cls} : \t{results_by_tag[cls]["strict"]["f1"]:.3f}')

        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    def predictold(self, input_string):
        self.eval()
        self.freeze()
        input_ids = self.tokenizer.encode(input_string, add_special_tokens=True)
        #print(input_ids)

        # convert to tensors & run the model_token_classification
        prepared_input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).to(self.device) # because batch_size = 1
        with torch.no_grad():
            output = self.model_token_classification(input_ids=prepared_input_ids, return_dict=True)
            logits = output["logits"]

        # extract results
        indices = torch.argmax(logits.detach().cpu(), dim=-1).squeeze(dim=0).tolist()  # reduce to [batch_size, seq_len] as list

        results = []
        for id, ind in zip(input_ids, indices):
            #print(f"\t[{self.tokenizer.decode(id)}] -> {self.bio2tag_list[ind]}")
            results.append({
                "token_id": id,
                "token": self.tokenizer.decode(id),
                "tag_id": ind,
                "tag": self.bio2tag_list[ind]
            })
        return results

    def predict(self, texts):
        self.eval()
        self.freeze()
        input_ids = self.tokenizer.encode(input_string, add_special_tokens=True)
        # print(input_ids)

        # convert to tensors & run the model_token_classification
        prepared_input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).to(self.device)  # because batch_size = 1
        with torch.no_grad():
            output = self.model_token_classification(input_ids=prepared_input_ids, return_dict=True)
            logits = output["logits"]

        # extract results
        indices = torch.argmax(logits.detach().cpu(), dim=-1).squeeze(
            dim=0).tolist()  # reduce to [batch_size, seq_len] as list

        results = []
        for id, ind in zip(input_ids, indices):
            # print(f"\t[{self.tokenizer.decode(id)}] -> {self.bio2tag_list[ind]}")
            results.append({
                "token_id": id,
                "token": self.tokenizer.decode(id),
                "tag_id": ind,
                "tag": self.bio2tag_list[ind]
            })
        return results

    def on_save_checkpoint(self, checkpoint) -> None:
        self.save()

    def save(self):
        folder = "trained_model"
        obj = {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "model_max_length": self.model_max_length,
            "bio2tag_list": self.bio2tag_list,
            "tag_list": self.tag_list
        }
        os.makedirs(folder + "/model_token_classification", exist_ok=True)
        os.makedirs(folder + "/tokenizer", exist_ok=True)
        self.model_token_classification.save_pretrained(save_directory=folder + "/model_token_classification")
        self.tokenizer.save_pretrained(save_directory=folder + "/tokenizer")
        torch.save(obj, folder+"/data.bin")
        print(f"Model saved in '{folder}'.")

    def load():
        folder = "trained_model"
        obj = torch.load(folder+"/data.bin")

        model = Model(
            model_name=folder+"/model",
            tokenizer_name=folder + "/tokenizer",
            model_max_length=obj["model_max_length"],
            bio2tag_list=obj["bio2tag_list"],
            tag_list=obj["tag_list"]
        )
        print(obj["bio2tag_list"])
        print(obj["tag_list"])
        model.eval()
        print("Model successfully loaded.")
        return model

    def set_device(self, device):
        self.to_device = device
        self.to(device)


class MyDataset(Dataset):
    def __init__(self, instances):
        self.instances = []

        # run check
        for instance in instances:
            ok = True
            if len(instance["ner_ids"]) != len(instance["tokens"]):
                print("Different length ner_tags found")
                ok = False
            else:
                for tag, token in zip(instance["ner_ids"], instance["tokens"]):
                    if token.strip() == "":
                        ok = False
                        print("Empty token found")
            if ok:
                self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]

class MyCollator(object):
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, input_batch):
        batch_input_ids, batch_labels, batch_attention, batch_token_idx = [], [], [], []
        max_len = 0

        for instance in input_batch:
            instance_ids, instance_labels, instance_attention, instance_token_idx = [], [], [], []

            for i in range(len(instance["tokens"])):
                subids = self.tokenizer.encode(instance["tokens"][i], add_special_tokens=False)
                sublabels = [instance["ner_ids"][i]]

                if len(subids) > 1:  # we have a word split in more than 1 subids, fill appropriately
                    filler_sublabel = sublabels[0] if sublabels[0] % 2 == 0 else sublabels[0] + 1
                    sublabels.extend([filler_sublabel] * (len(subids) - 1))

                instance_ids.extend(subids)  # extend with the number of subids
                instance_labels.extend(sublabels)  # extend with the number of subtags
                instance_token_idx.extend([i] * len(subids))  # extend with the id of the token

                assert len(subids) == len(sublabels) # check for possible errors in the dataset

            if len(instance_ids) != len(instance_labels):
                print(len(instance_ids))
                print(len(instance_labels))
                print(instance_ids)
                print(instance_labels)
            assert len(instance_ids) == len(instance_labels)

            # cut to max sequence length, if needed
            if len(instance_ids) > self.max_seq_len - 2:
                instance_ids = instance_ids[:self.max_seq_len - 2]
                instance_labels = instance_labels[:self.max_seq_len - 2]
                instance_token_idx = instance_token_idx[:self.max_seq_len - 2]

            # prepend and append special tokens, if needed
            #print()
            #print(instance_ids)
            if self.tokenizer.cls_token_id and self.tokenizer.sep_token_id:
                instance_ids = [self.tokenizer.cls_token_id] + instance_ids + [self.tokenizer.sep_token_id]
                instance_labels = [0] + instance_labels + [0]
                instance_token_idx = [-1] + instance_token_idx  # no need to pad the last, will do so automatically at return
            #print(instance_ids)
            instance_attention = [1] * len(instance_ids)


            # update max_len for later padding
            max_len = max(max_len, len(instance_ids))

            # add to batch
            batch_input_ids.append(torch.LongTensor(instance_ids))
            batch_labels.append(torch.LongTensor(instance_labels))
            batch_attention.append(torch.LongTensor(instance_attention))
            batch_token_idx.append(torch.LongTensor(instance_token_idx))

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True,
                                                         padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0),
            "token_idx": torch.nn.utils.rnn.pad_sequence(batch_token_idx, batch_first=True, padding_value=-1)
        }

def train_model(
        automodel_name: str,
        tokenizer_name: str,

        train_file: str = None,
        validation_file: str = None,
        test_file: str = None,

        gpus: int = 1,
        batch_size: int = 8,
        accumulate_grad_batches: int = 1,
        lr: float = 1e-5,
        model_max_length: int = 512
):
    print(f"Training with transformer model_token_classification / tokenizer: {automodel_name} / {tokenizer_name}")
    if train_file != "":
        print(f"\t with training file {train_file}")
    if validation_file != "":
        print(f"\t with validation file {validation_file}")
    if test_file != "":
        print(f"\t with test file {test_file}")

    if train_file == "" or validation_file == "" or test_file == "":
        print("\n A train/validation/test files must be given.")
        return

    print("\t batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(
        batch_size,
        accumulate_grad_batches,
        batch_size * accumulate_grad_batches)
    )

    # load data

    import random
    with open(train_file, "r", encoding="utf8") as f:
        train_data = json.load(f)[:200]
    with open(validation_file, "r", encoding="utf8") as f:
        validation_data = json.load(f)[:200]
    with open(test_file, "r", encoding="utf8") as f:
        test_data = json.load(f)[:200]
    train_data += test_data

    # deduce bio2 tag mapping and simple tag list, required by nervaluate
    tags = []  # tags without the B- or I- prefix
    bio2tags = ["*"]*31 # tags with the B- and I- prefix, all tags are here (30 + 1='O')
    for instance in train_data + validation_data + test_data:
        for tag, id in zip(instance["ner_tags"], instance["ner_ids"]):
            bio2tags[id] = tag

    print(f"Dataset contains {len(bio2tags)} BIO2 classes: {bio2tags}.")
    tags = sorted(list(set([tag[2:] if len(tag)>2 else tag for tag in bio2tags]))) # skip B- and I-
    print(f"\nThere are {len(tags)} classes: {tags}")

    # init tokenizer for collator, and start loading data
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, strip_accents=False)

    print("Loading data ...")
    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(validation_data)
    test_dataset = MyDataset(test_data)

    my_collator = MyCollator(tokenizer=tokenizer, max_seq_len=model_max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True,
                                  collate_fn=my_collator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=False,
                                collate_fn=my_collator, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False,
                                 collate_fn=my_collator, pin_memory=True)

    print("Train dataset has {} instances.".format(len(train_dataset)))
    print("Valid dataset has {} instances.".format(len(val_dataset)))
    print("Test dataset has {} instances.\n".format(len(test_dataset)))

    model = Model(
        model_name=automodel_name,
        lr=lr,
        model_max_length=model_max_length,
        bio2tag_list=list(bio2tags),
        tag_list=tags
    )

    early_stop = EarlyStopping(
        monitor='valid/strict',
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/strict",
        dirpath="model_token_classification",
        filename="NERModel",
        save_top_k=1,
        mode='max'
    )

    trainer = pl.Trainer(
        gpus=gpus,
        callbacks=[early_stop, checkpoint_callback],
        #limit_train_batches=10,
        #limit_val_batches=2,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    print("\nEvaluating model_token_classification on the VALIDATION dataset:")
    result_valid = trainer.test(model, val_dataloader)
    #print("\nEvaluating model_token_classification on the TEST dataset:")
    #result_test = trainer.test(model_token_classification, test_dataloader)

    print("\nDone training.\n")

    return model


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser() # todo redo defaults
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--model_name', type=str,
                        default="dumitrescustefan/bert-base-romanian-cased-v1")
    parser.add_argument('--tokenizer_name', type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--train_file", type=str, default="../data/test.json")
    parser.add_argument("--validation_file", type=str, default="../data/test.json")
    parser.add_argument("--test_file", type=str, default="../data/test.json")
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)
    parser.add_argument('--results_file', type=str, default="ronec_v2_results.json")

    args = parser.parse_args()

    if args.tokenizer_name == "":
        args.tokenizer_name = args.model_name

    # train model_token_classification, skip if already trained
    model = train_model(
        automodel_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
        gpus=args.gpus,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        lr=args.lr
    )


    # this is how to run
    device = "cuda" # or "cpu"
    model = Model.load()
    model.set_device(device)

    sentence = "Din 2017, când a început procesul de transformare a fabricii din Otopeni, până în prezent, Philip Morris International (PMI) a investit 500 de milioane de dolari pentru dezvoltarea capacităților de producție, formarea angajaților și implementarea unor soluții care vizează sustenabilitatea. Din această sumă, aproape 100 de milioane de dolari au fost investiți doar în 2021, iar în perioada 2022-2023, PMI va mai investi peste 100 de milioane de dolari."
    results = model.predict(sentence)

    for result in results:
        print(f"{result['token']: >16} = {result['tag']: <12} token_id={result['token_id']}, tag_id={result['tag_id']}")


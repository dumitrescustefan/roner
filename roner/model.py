# -*- coding: utf8 -*-

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm.autonotebook import tqdm as tqdm
import stanza

class NER():
    bio2tag_list = ['O', 'B-PERSON', 'I-PERSON', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-NAT_REL_POL',
     'I-NAT_REL_POL', 'B-EVENT', 'I-EVENT', 'B-LANGUAGE', 'I-LANGUAGE', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-DATETIME',
     'I-DATETIME', 'B-PERIOD', 'I-PERIOD', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-NUMERIC', 'I-NUMERIC',
     'B-ORDINAL', 'I-ORDINAL', 'B-FACILITY', 'I-FACILITY']

    def __init__(self,
                 model="dumitrescustefan/bert-base-romanian-ner",
                 use_gpu=True,
                 batch_size=4,
                 window_size=512,
                 num_workers=0,
                 named_persons_only=False,
                 verbose=False,
                 bio2tag_list=None):

        # overwrite tag lists, if given
        if bio2tag_list:
            NER.bio2tag_list = bio2tag_list
            if verbose:
                print(f"Overriding default bio2 tag list with {bio2tag_list}")

        # look for GPU if requested
        self.device = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                if verbose:
                    current_device_id = torch.cuda.current_device()
                    current_device_name = torch.cuda.get_device_name(current_device_id)
                    print(f"Found {torch.cuda.device_count()} GPU(s), using GPU #{current_device_id}: {current_device_name}")
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.batch_size = batch_size
        self.named_persons_only=named_persons_only
        self.window_size = window_size
        self.overlap_last = window_size//4
        self.num_workers = num_workers
        self.verbose = verbose

        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(model, strip_tokens=False, lower_case=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(NER.bio2tag_list))
        self.model.to(self.device)
        self.model.eval()

        try:
            self.stanza_nlp = stanza.Pipeline('ro', processors='tokenize,pos', use_gpu=use_gpu, logging_level='WARN')
        except:
            stanza.download('ro')
            self.stanza_nlp = stanza.Pipeline('ro', processors='tokenize,pos', use_gpu=use_gpu, logging_level='WARN')

    def __call__(self, texts:[]):
        """

        :param texts:
        :return:
        """

        # pre-flight checks
        if not isinstance(texts, list):
            if isinstance(texts, str):
                texts = [texts]
            else:
                raise Exception(f"Input type not supported {type(texts)}. Please input a string or a list of strings.")

        # setup dataloader
        dataset = NER.InferenceDataset(texts=texts, tokenizer=self.tokenizer, stanza=self.stanza_nlp, model_size=self.window_size, overlap_last=self.overlap_last, named_persons_only=self.named_persons_only, verbose=self.verbose)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=NER.InferenceCollator(model_size=self.window_size, tokenizer=self.tokenizer, device=self.device))

        # run prediction
        with torch.no_grad():
                for batch in tqdm(dataloader, desc="Running NER", unit="texts", disable=not self.verbose):
                    output = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True
                    )
                    logits = output["logits"] # [batch_size, model_size, bio2label_size]
                    indices = torch.argmax(logits.cpu(), dim=-1).squeeze(dim=-1).tolist()  # reduce to [batch_size, model_size] as list
                    dataset.predictions.extend(indices)

        # agreement between overlapping windows
        dataset.run_agreement()

        return dataset.processed_texts

    def detokenize(self, outputs:list):
        """
        Use this to detokenize the text back to its original form
        :param outputs: list of dicts generated by running NER on text
        :return: list of strings representing the detokenized text
        """
        texts = []
        for output in outputs:
            text = output["span_before"]
            for word in output["words"]:
                text += word['text'] + word['span_after']
            texts.append(text)
        return texts

    class InferenceDataset(Dataset):
        def __init__(self, texts: [], tokenizer: str, stanza, model_size:int, overlap_last:int, named_persons_only:bool, verbose:bool):
            self.model_size = model_size
            self.overlap_last = overlap_last
            self.named_persons_only = named_persons_only
            self.verbose = verbose
            self.processed_texts = [] # list of tokenized texts
            self.instances = []
            self.predictions = []
            self.tokenizer = tokenizer

            for text_id, text in enumerate(tqdm(texts, desc="Preprocessing texts", unit="texts", disable=not self.verbose)):
                units, input_ids = [], []

                # tokenize/pos text
                doc = stanza(text)
                for sentence in doc.sentences:
                    for i in range(len(sentence.words)):
                        word = sentence.words[i]

                        # get span_after, required for detokenization
                        next_word = None
                        if i+1 < len(sentence.words):
                            next_word = sentence.words[i+1]
                        span_after = ''
                        if word.end_char<len(text):
                            if next_word:
                                span_after = text[word.end_char:next_word.start_char]
                            else:
                                span_after = text[word.end_char:]

                        units.append(
                            {
                                "text": word.text,
                                "pos": word.upos,
                                "start_char": word.start_char,
                                "end_char": word.end_char,
                                "token_ids": tokenizer(word.text, add_special_tokens=False)['input_ids'],
                                "tag": None,
                                "span_after": span_after
                            }
                        )
                        input_ids.extend(units[-1]["token_ids"])

                # get span_before for possible detokenization
                span_before = text
                if len(units)>0:
                    span_before = text[0:units[0]["start_char"]] # text that's not a word, preceding the first word

                # save processed element
                self.processed_texts.append({
                    "text": text,
                    "input_ids": input_ids,
                    "span_before": span_before,
                    "words": units
                })

                # generate instances
                for i in range(0, len(input_ids), model_size-overlap_last-2):
                    start = i
                    end = min(i+model_size-2, len(input_ids))
                    self.instances.append({
                        "id": text_id,
                        "start": start,
                        "end": end,
                        "input_ids": input_ids[start:end],
                    })
                    #print(f"text_id {text_id}: {start}-{end}")
            #print("Done instances")

        def run_agreement(self):
            assert len(self.instances)==len(self.predictions)
            current_text_idx = -1
            while current_text_idx<len(self.processed_texts)-1:
                current_text_idx+=1
                #print(f"Agreement for text idx {current_text_idx}")

                # get agreement list
                agreement = []
                for _ in range(len(self.processed_texts[current_text_idx]["input_ids"])): # create cell holders
                    agreement.append([])
                for i in range(len(self.instances)):
                    if self.instances[i]["id"] == current_text_idx:
                        instance = self.instances[i]
                        prediction = self.predictions[i][1:-1]
                        for j in range(len(instance["input_ids"])):
                            abs_pos = instance["start"]+j
                            agreement_cell = {
                                "input_id": instance["input_ids"][j],
                                "abs_pos": abs_pos,
                                "rel_pos": j,
                                "tag": prediction[j]
                            }
                            agreement[abs_pos].append(agreement_cell)

                # solve disagreements
                tags = []
                mid_point = self.model_size-int(self.overlap_last/2)
                for i in range(len(agreement)):
                    if len(agreement[i])==1:
                        tags.append(agreement[i][0]['tag'])
                    else:
                        if agreement[i][0]['tag'] == agreement[i][1]['tag']:
                            tags.append(agreement[i][0]['tag'])
                        else: # disagreement
                            #print(f"Disagreement between:\n{agreement[i][0]}\nand\n{agreement[i][1]}")
                            #print(self.tokenizer.decode(self.processed_texts[current_text_idx]["input_ids"][i-10:i+1]))

                            if agreement[i][0]['rel_pos']<mid_point:
                                tags.append(agreement[i][0]['tag'])
                                #print(f"\t prefer: left")
                            else:
                                tags.append(agreement[i][1]['tag'])
                                #print(f"\t prefer: right")
                            #print(tags[-10:])
                            #print("----")

                # save agreement and mark multi-word entities
                j = 0
                for i in range(len(self.processed_texts[current_text_idx]["words"])):
                    cnt = len(self.processed_texts[current_text_idx]["words"][i]["token_ids"])

                    if cnt == 0: # bug when transformer tokenizer gives no token ids for this word, and j==len(sequence token_ids)
                        tag = 'O'
                        self.processed_texts[current_text_idx]["words"][i]["tag_ids"] = [0] # fake tag added
                    else:
                        tag = NER.bio2tag_list[tags[j]]
                        self.processed_texts[current_text_idx]["words"][i]["tag_ids"] = tags[j:j + cnt]  # save tags

                    if tag != "O":
                        tag = tag[2:]
                        if tag == "PERSON" and self.named_persons_only is True and self.processed_texts[current_text_idx]["words"][i]['pos'] != "PROPN":
                            tag = "O"
                    self.processed_texts[current_text_idx]["words"][i]["tag"] = tag
                    j += cnt

                    mwe = False
                    if i>0:
                        if self.processed_texts[current_text_idx]["words"][i]["tag_ids"][0]>0 and self.processed_texts[current_text_idx]["words"][i]["tag_ids"][0]%2==0:
                            if self.processed_texts[current_text_idx]["words"][i-1]["tag"] == self.processed_texts[current_text_idx]["words"][i]["tag"]:
                                mwe = True
                    self.processed_texts[current_text_idx]["words"][i]["multi_word_entity"] = mwe

        def __len__(self):
            return len(self.instances)

        def __getitem__(self, i):
            return self.instances[i]

    class InferenceCollator(object):
        def __init__(self, model_size, tokenizer, device):
            self.model_size = model_size
            self.tokenizer = tokenizer
            self.device = device

        def __call__(self, input_batch):
            """
                input_batch is a list of instances
                return a dict of padded tensors
            """
            #[{'id': 0, 'start': 0, 'end': 16, 'input_ids': [1138, 643, 5263, 2099, 2934, 372, 1452, 18, 6223, 18, 2703, 13365, 1939, 392, 908, 27]}]

            batch_input_ids, batch_attention, batch_text_idx, batch_start, batch_end = [], [], [], [], []

            for instance in input_batch:
                batch_text_idx.append(instance["id"]) # save id of text
                batch_start.append(instance["start"])
                batch_end.append(instance["end"])
                instance_ids = instance["input_ids"]

                if self.tokenizer.cls_token_id and self.tokenizer.sep_token_id: # prepend and append special tokens, if needed
                    instance_ids = [self.tokenizer.cls_token_id] + instance_ids + [self.tokenizer.sep_token_id]
                instance_attention = [1] * len(instance_ids)

                # add to batch
                batch_input_ids.append(torch.LongTensor(instance_ids))
                batch_attention.append(torch.LongTensor(instance_attention))

            return {
                "input_ids": torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True,
                                                             padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0).to(self.device),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(batch_attention, batch_first=True, padding_value=0).to(self.device),
                "text_idx": batch_text_idx,
                "start": batch_start,
                "end": batch_end
            }
)
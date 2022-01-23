## !!! This is a work-in-progress, will be updated soon !!!

# RoNER

RoNER is a Named Entity Recognition model based on a pre-trained [BERT transformer model](https://huggingface.co/dumitrescustefan/bert-base-romanian-ner) trained on [RONECv2](https://github.com/dumitrescustefan/ronec). It is meant to be an easy to use, high-accuracy Python package providing Romanian NER.

RoNER handles _text splitting_, _word-to-subword alignment_, and it works with arbitrarily _long text sequences_ on CPU or GPU.  



## Instalation & usage

Install with: ``pip install roner``

Run with:
```python
import roner
ner = roner.NER()

input_texts = ["George merge cu trenul Cluj - Timișoara de ora 6:20.", 
               "Grecia are capitala la Atena."]

output_texts = ner(input_texts)

for output_text in output_texts:
  print(f"Original text: {output_text['text']}")
  for word in output_text['words']:
    print(f"{word['text']:>20} = {word['tag']}")
```

#### RoNEC input

RoNER accepts either strings or lists of strings as input. If you pass a single string, it will convert it to a list containing this string.

#### RoNEC output

RoNER outputs a list of dictionary objects corresponding to the given input list of strings. A dictionary entry consists of:

```json
{
  "text": <<original text given as input>>,
  "input_ids": <<token ids of the original text>>,
  "words": [{
      "text": <<each word>>,
      "tag": <<entity label>>
      "pos": <<part of speech of this word>>,
      "multi_word_entity": <<True if this word is linked to the previous one>>,
      "span_after": <<span of text linking this word to the next>>,
      "start_char": <<start position of this word in the original text>>,
      "end_char": <<end position of this word in the original text>>,
      "token_ids": <<list of subtoken ids as given by the BERT tokenizer>>,
      "tag_ids": <<list of BIO2 tags assigned by the model for each subtoken>>
    }]
}
```

This information is sufficient to save word-to-subtoken alignment, to have access to the original text as well as having other usable info such as the start and end positions for each word.

To list entities, simply iterate over all the words in the dict, printing the word itself ``word['text']`` and its label ``word['tag']``.

## RoNER properties and considerations


#### Constructor options

The NER constructor has the following properties:

* ``model:str`` Override this if you want to use your own pretrained model. Specify either a HuggingFace model or a folder location. If you use a different tag set than RONECv2, you need to also override the ``bio2tag_list`` option. The default model is ``dumitrescustefan/bert-base-romanian-ner``
* ``use_gpu:bool`` Set to True if you want to use the GPU (much faster!). Default is enabled; if there is no GPU found, it falls back to CPU.
* ``batch_size:int`` How many sequences to process in parallel. On an 11GB GPU you can use batch_size = 8. Default is 4. Larger values mean faster processing - increase until you get OOM errors.
* ``window_size:int`` Model size. BERT uses by default 512. Change if you know what you're doing. RoNER uses this value to compute overlapping windows (will overlap last quarter of the window).
* ``num_workers:int`` How many workers to use for feeding data to GPU/CPU. Default is 0, meaning use the main process for data loading. Safest option is to leave at 0 to avoid possible errors at forking on different OSes.
* ``named_persons_only:bool`` Set to True to output only named persons labeled with the class PERSON. This parameter is further explained below. 
* ``verbose:bool`` Set to True to get processing info. Leave it at its default False value for peace and quiet.
* ``bio2tag_list:list`` Default None, change only if you trained your own model with different ordering of the BIO2 tags.

#### Implicit tokenization of texts

Please note that RoNER uses Stanza to handle Romanian tokenization into words and part-of-speech tagging. On first run, it will download not only the NER transformer model, but also Stanza's Romanian data package.

#### 'PERSON' class handling

An important aspect that requires clarification is the handling of the ``PERSON`` label. In RONECv2, persons are not only names of persons (proper nouns, aka ``George Mihailescu``), but also any common noun that refers to a person, such as ``ea``, ``fratele`` or ``doctorul``. For applications that do not need to handle this scenario, please set the ``named_persons_only`` value to ``True`` in RoNER's constructor. 

What this does is use the part of speech tagging provided by Stanza and only set as ``PERSON``s proper nouns.

#### Multi-word entities

Sometimes, entities span multiple words. To handle this, RoNER has a special property named ``multi_word_entity``, which, when True, means that the current entity is linked to the previous one. Single-word entities will have this property set to False, as will the _first_ word of multi-word entities. This is necessary to distinguish between sequential multi-word entities. 

#### Detokenization

One particular use-case for a NER is to perform text anonymization, which means to replace entities with their label. With this in mind, RoNER has a ``detokenization`` function, that, applied to the outputs, will recreate the original strings. 

To perform the anonymization, iterate through all the words, and replace the word's text with its label as in ``word['text'] = word['tag']``.
Then, simply run ``anonymized_texts = ner.detokenize(outputs)``. This will preserve spaces, new-lines and other characters.

#### NER accuracy metrics

Finally, because we trained the model on a modified version of RONECv2 (we performed data augumentation on the sentences, used a different training scheme and other train/validation/test splits) we are unable to compare to the standard baseline of RONECv2 as part of the original test set is now included in our training data, but we have obtained, to our knowledge, SOTA results on Romanian. This repo is meant to be used in production, and not for comparisons to other models.

## BibTeX entry and citation info

Please consider citing the following [paper](https://arxiv.org/abs/1909.01247) as a thank you to the authors of the RONEC, even if it describes v1 of the corpus and you are using a model trained on v2 by the same authors: 
```
Dumitrescu, Stefan Daniel, and Andrei-Marius Avram. "Introducing RONEC--the Romanian Named Entity Corpus." arXiv preprint arXiv:1909.01247 (2019).
```
or in .bibtex format:
```
@article{dumitrescu2019introducing,
  title={Introducing RONEC--the Romanian Named Entity Corpus},
  author={Dumitrescu, Stefan Daniel and Avram, Andrei-Marius},
  journal={arXiv preprint arXiv:1909.01247},
  year={2019}
}
```



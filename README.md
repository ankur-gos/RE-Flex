# RE-Flex

RE-Flex is a simple to use framework to perform relation extraction using the contextualized representations produced 
from masked language models such as BERT and RoBERTa. Instead of performing forward passes through the language model to
generate answers to relational queries, RE-Flex matches the contextual representations of the prediction to words in the
associated context, and predicts the most likely word in this context.




## Requirements

1. Pytorch 1.2.0
2. torchvision 0.4.0
2. Python 3.6

Note that updated versions of these requirements may work, but these were the versions developed and tested with.

## Installation

Clone this repository. After cloning, you can install via pip:

```
   pip install .
```

Next, download the model weights. You can find them [here](www.example.com).
Behind the scenes, RE-Flex utilizes fairseq, and more specifically the RoBERTa model interface.
The linked weights are the large version of 
[RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md),
so if you have memory requirements, you can
alternatively use [roberta.base](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)
weights instead.

Finally, make sure to install a spacy model. We recommend en_core_web_lg.

```
    python -m spacy download en_core_web_lg
```

## Getting started

We'll get started with a single example. We can test an example extraction
using a built in extraction script.

We'll extract what team Giannis Antetokounmpo plays for.

Following previous work in general relation extraction, relations are defined by *relational templates*,
natural language representations of the relation. In our case, we will use cloze versions of the templates.
For example, for a relation that defines the team a player plays for, we might define
the following template:

```
[X] plays for the [Y].
```

We use this template to extract what team Giannis plays for:

```
python -m reflex.scripts.infer_one \
    --context "Giannis Sina Ugo Antetokounmpo is a Greek professional basketball player for the Milwaukee Bucks of the National Basketball Association. Born in Greece to Nigerian parents, Antetokounmpo began playing basketball for the youth teams of Filathlitikos in Athens." 
    --entity "Giannis Antetokounmpo" \
    --template "[X] plays for the [Y]"

> loading archive file ./roberta_large/
> | dictionary: 50264 types
> Bucks
```

By default RE-Flex returns a single token from the input context. Alternatively,
you can expand the response by adding the --expand flag:

```
python -m reflex.scripts.infer_one \
    --context "Giannis Sina Ugo Antetokounmpo is a Greek professional basketball player for the Milwaukee Bucks of the National Basketball Association. Born in Greece to Nigerian parents, Antetokounmpo began playing basketball for the youth teams of Filathlitikos in Athens." 
    --entity "Giannis Antetokounmpo" \
    --template "[X] plays for the [Y]" \
    --expand

> loading archive file ./roberta_large/
> | dictionary: 50264 types
> the Milwaukee Bucks
```

## Performing relation extraction

RE-Flex focuses on slot-filling based relation extraction, where given
a set of head entities, a set of contexts associated with those head
entities, and a relational template, a set of corresponding tail entities
can be inferred.

Take a look at [extract_plays_for.ipynb]() for an example of extracting
a set of teams that sports players play for.







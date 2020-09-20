# BIDAF Implementation

This folder is the BiDAF implementation used in the RE-Flex paper for the reported experiments. It is derived from [zhuzhicai's implementation](https://github.com/zhuzhicai/SQuAD2.0-Baseline-Test-with-BiDAF-No-Answer), which is itself derived
from the original BiDAF implementation. I use the trained Squad weights that match the expected performance of BiDAF on SQuAD.

It is slightly updated to be able to take the specific inputs of the datasets.

This is best run in a docker container. It has been built and tested with docker version 19.03.1. To build this container, first download Glove weights to this directory. They can be downloaded from [here](https://drive.google.com/drive/folders/18D9cqpKHz_F1VEXtZ1qpgPpTnV2NeDEi?usp=sharing).

Then run:

```
   docker build -t TAG .
```

If you are building to reproduce experiments, make sure that in RE-Flex/reflex/docker-compose-bidaf.yml that you update the image to the TAG you used above.


The main entry points for experiments that force answers is in [run.sh](https://github.com/ankur-gos/RE-Flex/blob/master/reflex/bidaf/run.sh), and the main entry points for experiments that allow no answer is in [run2.sh](https://github.com/ankur-gos/RE-Flex/blob/master/reflex/bidaf/run2.sh).


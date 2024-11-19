
SBERT BI/CROSS PUZZLES IR
--
Ellis Fitzgerald - COS470 - Project 2

How to run:
-

There are two (2) ways to run.

Fine-tuning locally:

        C:>python main.py path/to/answers path/to/topic1...path/to/topicN -q path/to/qrel1 

The arguments in order:
- Positional: Corpus JSON File (Answers)
- Positional: N Topic JSON Files (Queries)
- Optional ( **-q** ): TREC QREL TSV (1)

The qrel to be split for train, eval, and testing.
this will grant N * 4 result files. With N representing the topic file count, and 4 representing a respective result for Pretrained Bi-Encoder, Pretrained Cross-Encoder, Fine-tuned Bi-Encoder, and Fine-tuned Cross-Encoder

Result file generation only:

        C:>python main.py path/to/answers path/to/topic1...path/to/topicN -qs path/to/train path/to/eval path/to/test -bc path/to/bi/checkpoint -cc path/to/ce/checkpoint

- Positional: Corpus JSON File (Answers)
- Positional: N Topic JSON Files (Queries)
- Optional ( **-qs** ): TREC QREL TSV's: Train, Evaluation, Test
- Optional ( **-bc** ): HuggingFace Checkpoint to local Bi-Encoder model
- Optional ( **-cc** ): HuggingFace Checkpoint to local Cross-Encoder model

The ( **-qs** ) argument can only be supplied after fine-tuning as it MUST be consistent with the splits used for training. Otherwise your results will be invalid!

For help:
        
        C:>python main.py -h

*Have fun*

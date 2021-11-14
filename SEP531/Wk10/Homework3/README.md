# Homework 3

The goal of this assignment is to translate a given source sentence (e.g., German) to a target sentence (e.g., English). You need to implement a Neural Machine Translation (NMT) system using a sequence-to-sequence (Seq2Seq) model with global attention mechansim (i.e., Luong attention'15). Please complete the blank code correctly. 
Specifically, I plan to list all students' BLEU scores and rank them in the highest order.

When grading your assignments, we will be doing it through your model's BLEU score on the test set. Specifically, we plan to enumerate all students' BLEU scores from their model and rank them in the highest order. So, if you want to get the high score, you have to improve your NMT system by applying other techniques (e.g., teacher forcing, stack more layers, other attention mechansim). If you have applied other methods to improve performance, please write a report with the reason. 

## Usage
### Downloading the dataset
After completing all code, you can train your model using IWSLT 2014 De-En dataset as training data. The source and the target languages are German and English, respectively. Before training your model, you must download the IWSLT 2014 De-En dataset by excuting the following command:

```bash
wget http://www.cs.cmu.edu/~pengchey/iwslt2014_ende.zip
unzip iwslt2014_ende.zip
```

### Building the vocabulary
After downloading the dataset, please execute the following command to generate the vocabulary file for both source and target languages:
```bash
bash run.sh vocab [options]
```
Note that you can change parameters or arguments to improve the quality of vocabulary.

### Training your model
After building the vocabulary file, please excute the following command to train your model:
```bash
bash run.sh train [options]
```
Also, you can change parameters to obtain better performance on dev set.

### Testing your model
After completing the training process, please execute the following command to obtain the BLEU score and generated translations via a beam search from your trained model:
```bash
bash run.sh test [options]
```

## Submission

Please submit your code and report to KLMS. We will make a site for submissions of HW3. Your submission should include the following:

- Source code
- Checkpoint corresponding to your trained model
- Report (must be **pdf format** with **'이름_학번.zip'**)

Your report should include the following to get credits:

- Explanation of your model that you implemented
    - *Note: if you apply other techniques to improve better performance than the baseline (i.e., Seq2Seq with global attention mechanism), then you also have to explain the improved version of your model*
- Analysis of your results
    - Quantitative comparisons between several experiments in terms of BLEU score
    - *(Optional) Visualize your attention plot*

## Submission due

**Due date: 11/23 (Tue), 23.59 p.m.**

## References

### Code
- [https://github.com/pcyin/pytorch_nmt](https://github.com/pcyin/pytorch_nmt)
- [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a4.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a4.pdf)
### Papers
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) (Luong Attention)

## Contact

If you have some questions, please contact to me via an email. (passing2961@gmail.com)




Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence

NIST: Based off BLEU, NIST additionally looks at the amount of n-gram overlap between strings while weighting for frequency and information gain.

BLEU: One of the first metric in machine translation, calculates similarity by number of matching n-gram.

WER (Word Error Rate): Reviews how many deletions, insertions, and substitutions are necessary before two strings are similar.

LCS (Longest Common Substring): As the name implies, given two strings it finds the length of the most common substring.

ED (Edit Distance): Computes 'n' edits required to transform the strings being compared to one another.

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

| Metric    | Train  | Dev    | Test   |
|-----------|--------|--------|--------|
| NIST      | 0.492  | 0.593  | 0.462  |
| BLEU      | 0.370  | 0.433  | 0.352  |
| WER       | -0.360 | -0.452 | -0.361 |
| LCS       | 0.462  | 0.468  | 0.504  |
| Edit Dist | 0.033  | -0.175 | -0.038 |

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).


## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.

Example usage:
`python sys_pearson.py --sts_data stsbenchmark/sts-dev.csv`,
`python sys_pearson.py --sts_data stsbenchmark/sts-test.csv`,
`python sys_pearson.py --sts_data stsbenchmark/sts-train.csv`
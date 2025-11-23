# %%

#Imports and Setup

'''
#OS is for talking to the operating system. Why do we need it? Mostly for os.environ which was used to handle errors (first block of code here)
#json is for HF dataset work. Those datasets work well with json.
#re is used a lot for preprocessing (stripping HTML tags, normalizing whitespaces, etc)
#numpy and pandas: arrays and dataframes
'''
import os, json, re, numpy as np, pandas as pd

#torch import explanation
'''
#torch is PyTorch. Its a deep learning framework. Its the thing that actually runs our model. AutoModelForSequenceClassifier is build on top of PyTorch.
#Commands like "model.to(DEVICE)" or "torch.tensor(blablabla)" use PyTorch tensors instead of numpy arrays (tensors explanation provided way below in code)
#torch.tensor() converts Python lists or NumPy arrays into tensors that are better suited for model to use.
#torch.cuda.is_available() checks if GPU is available
#model.to(DEVICE) moves model and data to GPU or CPU
#torch.no_grad() tells PyTorch to not calculate gradients
#Everything that touches our model or GPU relies on the torch import
'''

#warnings is a standard PYTHON library that is meant to silence warning messages, helps output look cleaner
import torch, warnings

#typing is just a tool that helps with hinting in python. Helps in knowing inputs and outputs and what is expected where
from typing import List, Dict, Any, Tuple

#collections is a python module, counter is a subclass that helps with counting things. We use it mainly for label distribution counting.
from collections import Counter

#train_test_split import explanation
'''
#sklearn is a ML library. train_test_split is a function thats used to split data into training and testing.
#Function has parameters:
#test_size is the fraction you want to use for testing (our case is 10).
#train_size is the same for training (80 for us)
#random_state is a seed number to reproduce randomness (we used 42)
#stratify keeps label proportions equal, in our case we use stratify = labelset
'''
from sklearn.model_selection import train_test_split

#this comes from sklearn's metrics module which provides the metrics we can use to evaluate our model.
from sklearn.metrics import f1_score, precision_score, recall_score

#transformers and AutoTokenizer/AutoModelForSequenceClassification imports explanation
'''
#What is transformers? a Hugging Face library for ML tasks.
#Has pretrained transformer models (like BERT, RoBERTa, GPT, etc)
#Has Tokenizers that match each model's vocab and text preprocessing rules
#Transformers are built around the transformer architecture, which relies entirely on self attention
#(meaning that each token in a sequence "attends" to every other token in that sequence for context)
#This library covers nearly all modern NLP tasks. Using this means that we can grab and use models without coding them

#AutoTokenizer is a function that downloads/loads the exact tokenizer that matches our model (vocab, normalization, special tokens, all these things are taken  into account)
#Converts raw text to tensors ready to put into our model (input IDs, attention masks, token type IDs)

#What does autotokenizer do in more detail? (BTW this is done in the last line of code in this cell) {
  #When we call AutoTokenizer.from_pretrained(MODEL_NAME), it will first fetch the config and vocab from the model repo.
  #then it will make an instance of the right tokenizer class (for privBERT it will be a BERT style fast tokenizer)
  #it will make special tokens and give them the correct IDs: [PAD],[UNK],[MASK], etc
  #then it will apply model specific normalization.
  #How does text become token? First data is split by whitespaces and punctuation
  #rare words are split (regulation -> reg, ##ulation)
  #Special tokens are inserted (explanation for each will be provided later)
  #Truncation and Padding up to max length (BERT's hard limit is 512)
}


#What are the inputs for AutoTokenizer? {
  enc = tokenizer(
    List[str], #One or many sentences to be tokenized
    padding='max_length', #Pad to our max length?
    truncation=True, #Truncate or Not
    max_length=256, #Max number of tokens
    return_tensors="pt" #pt or tf (PyTorch TensorFlow) tensors
)
  #other inputs can include: add_special_tokens (is a boolean to add or not)
  #return_attention_mask (bool that is defaulted to true)
  #is_split_into_words (bool, use it if you split words manually, we dont do that so we dont use it)
}
#What are the outputs for AutoTokenizer? {
  #if the command is enc = tokenizer(blablabla), we will get a dictionary of:
  #input IDs: integer IDs for tokens
  #attention_mask: (1 for real, 0 for pad)
  #token_type_ids: 0 or 1 depending on wether or not the model uses token type IDs (our case will be 1 cuz bert uses them)
}

#attention masks explanation: {
  say our sentence is "We collect data" and our fixed length is 5.
  #our tokens will be [We, Collect, Data] but the input to the model will be padded to our fixed length and will look like this [We, Collect, Data, [empty], [empty]]
  #each token should have an input ID, so the input will look more like [21,22,43,0,0]. Attention masks will look like [1,1,1,0,0]. A binary representation showing where the padded ones are
  #why do we need it? we want the model to avoid attneding to the masks and using them as context. We want only actual data to be used as context.
}
#
#How  about AutoModelForSequenceClassification? What does it do? {
  #It loads the transformer backbone (like the base for BERT) with pretrained weights
  #attaches a classification head (further explanation)
  #MultiClass or MultiLabel? We are MultiLabel, so we use sigmoids, else it would use SoftMax
  #from_pretrained pulls weights, tokenizer, config that the og model uses. This makes sure that all this code can work on any device
}
#Inputs? We can feed the entire dictionary returned by the tokenizer straight into the model.
#Outputs? Will return a dictionary with certain attributes, looks like this:
{
  'loss': tensor(0.3452),     #only if labels were given
  'logits': tensor([[ 1.23, -0.56, 0.41, 2.10, -1.33]])
}
'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification #Loading Transformer from Hugging Face (loads a pretrained model)

#Disable W&B (they caused error earlier) and also use the warnings library to ignore future warnings from transformers
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


#Check if cuda supported GPU is available, else use cpu
#Use the torch library, one of its commands checks if you have a GPU or not
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#PrivBERT insertion
#MODEL_NAME = 'mukund/privbert'

#Dictionary of Labels (0 -> Collection, 1 -> Usage, etc)
id2label = {0:'Data Collection', 1:'Data Usage', 2:'Data Sharing', 3:'User Control', 4:'Other'}

#Opposite of last line (Collection -> 0, Usage -> 1, etc)
label2id = {v:k for k,v in id2label.items()}

#Number of labels present
NUM_LABELS = len(id2label)

'''
#Loads the tokenizer available from mukund/privbert. Download is done from Hugging Face.
#'from_pretrained(MODEL_NAME)' makes sure that tokenization is compatible with model weights later loaded
#IMPORTANT: Tokenizer converts raw text to Token IDs, attention masks, and token type IDs (if type IDs are used)
#Makes sure that the same vocab and preprocessing rules are used as the one that the model expects.
'''

MODEL_PATH =  "privbert_final"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


from pathlib import Path

# %%
#Load & Merge Files (Will be simpler in the future because we will have one file only)
'''
Workflow for this part:

#Provide data sources in an array of strings
#Try to detect exact text amd label column names, if failed try similar names
'''
BASE_DIR = Path(__file__).resolve().parent
#Data Sources
CANDIDATES = [
    BASE_DIR / 'pp_sentences_first5_Labelled.xlsx',
    BASE_DIR / 'pp_sentences_next40_label_.xlsx',
    BASE_DIR / 'pp_sentences_next55_label.xlsx'
]

#add more paths when needed
EXTRA_FILES = []

#Function to find the correct column. Takes dataframe and column candidate names as input. Returns EXACT column names if found
def detect_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}

    #Find EXACT columns. For example, if 'text' is found exactly, this for loop will return the column with the name 'text'
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]

    #Find SIMILAR columns. For example, if looking for 'sentences' and there exists a column name 'sentences_text', that will be returned as it is a substring.
    for c in df.columns:
        if any(name.lower() in c.lower() for name in candidates):
            return c
    return None


#Function to go through labels. Accepts Multiple Types: [0,2], "0,2", "0;2", "Data Collection; Data Sharing"
def parse_labels(val):

    #If value is list tuple or array, return it sorted.
    if isinstance(val, (list, tuple, np.ndarray)):
        return sorted({int(x) for x in val})

    #If missing value, empty list.
    if pd.isna(val):
        return []

    #Trim whitespaces
    s = str(val).strip()

    #Split on commas or semicolons.
    parts = re.split(r'[;,]\s*', s)


    #Basically, if all labels are digits, treat as an integer label id. Else, map it as name.
    #If mapping is found (Data Usage -> 1), append the digit to ids.
    #Finally, return sorted UNIQUE IDs
    ids = []
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r'\d+', p):
            ids.append(int(p))
        else:
            lid = label2id.get(p)
            if lid is None:
                for name, idx in label2id.items():
                    if p.lower() == name.lower():
                        lid = idx; break
                if lid is None:
                    matches = [idx for name, idx in label2id.items() if name.lower() in p.lower()]
                    if matches: lid = matches[0]
            if lid is not None:
                ids.append(lid)
    return sorted(set(ids))
#Finish Parse Labels Function


#Function that allows reading of both csv and excel (in case mixing happens. i.e: we have 5 data sources with 3 excel & 2 csv)
def read_any(path: str) -> pd.DataFrame:
    path = str(path)
    return pd.read_excel(path) if path.lower().endswith('.xlsx') else pd.read_csv(path)

#Function that does everything. Loops over paths, reads each file, detects text and label columns, normalize labels into set of sorted IDs, then merge everything into one DataFrame.
def load_and_merge(files):
    frames = []
    for p in files:

        #Only attempt on existing paths, avoids loading errors
        if os.path.exists(p):
            try:

                #Loads either excel or csv (recall _read_any function)
                dfp = read_any(p)

                #Find the best candidate for column that stores textual data (recall _detect_col function)
                tcol = detect_col(dfp, ['text','sentence','policy_text','segment'])

                #Find the best candidate for column that stores label data (recall _detect_col function)
                lcol = detect_col(dfp, ['labels','label','category','categories'])

                #If not found, show which part is skipped due to missing columns. Also show columns found.
                if tcol is None or lcol is None:
                    print(f'Skipping {p} — missing text/labels columns. Found: {list(dfp.columns)}')
                    continue

                #Rename columns to 'text' and 'labels' IMPORTANT
                dfp = dfp[[tcol,lcol]].rename(columns={tcol:'text', lcol:'labels'})

                #Parse Labels on 'labels' column and normalize labels IMPORTANT (if asked 'where do you normalize your data?')
                dfp['labels'] = dfp['labels'].apply(parse_labels)
                #Append normalized labels
                frames.append(dfp)
                print(f'Successfully Loaded {len(dfp)} rows from {p}') #Success message of how many rows of labels loaded.

            #Error message
            except Exception as e:
                print(f'Failed to parse {p}:', e)
    #More error handling
    if not frames:
        raise ValueError('No valid files found.')

    #Merging everything and returning the final merged Dataframe.
    merged = pd.concat(frames, ignore_index=True)
    return merged

#Builds list of files to load and merges everything together
files_to_load = [p for p in CANDIDATES if os.path.exists(p)] + EXTRA_FILES
merged = load_and_merge(files_to_load)

#Clean text function
def normalize_text(s: str) -> str:

    #Ensure it is a string and trim leading/trailing whitespaces
    s = str(s).strip()

    #If there is any sequence of repeated whitespace, collapse into one single space then return final string
    s = re.sub(r'\s+', ' ', s)
    return s

#First ensure that text is string then normalize each row
merged['text'] = merged['text'].astype(str).apply(normalize_text)

#Drop Empty Rows if text is empty after normalization
merged = merged[merged['text'].str.len() > 0].reset_index(drop=True)



# Drop rows that have no labels (unlabeled samples)
empty = merged['labels'].apply(len) == 0 #Checks if length of list of labels for each sample = 0 and makes a Bool List of yes or no (==0 or !==0)
if empty.any(): #drops only empty rows (merged[~empty] basically inverts the list made earlier and keeps anything that is true)
    print(f"Dropping {empty.sum()} unlabeled rows (no labels assigned).")
    merged = merged[~empty].reset_index(drop=True) #drop = True is for resetting index not dropping rows!


#If duplicate is found, merge into one and if any extra labels are applied, also merge
#(For example, if two texts are 'We use your data and share it', but they have labels [1], and [1,2]. Take the first and append the stuff present in the second but not the first)
merged = merged.groupby('text', as_index=False)['labels'].apply(lambda col: sorted({x for row in col for x in row}))


#Report label distribution
def label_stats(df):
    counts = Counter()
    for labs in df['labels']:
        for j in labs: counts[j]+=1
    total = len(df)
    print(f'\nDataset size: {total} rows')
    for j, name in id2label.items():
        print(f'  {j} ({name}): {counts[j]} occurrences')
    combos = Counter(tuple(l) for l in df['labels']).most_common(8)
    print('Top label combinations:', combos)

label_stats(merged)

df = merged.copy().reset_index(drop=True)
df.head()

# %%
#Binarize and Tokenize

'''
binarize_labels is a Helper function that converts our label lists into binary lists
(0,1,2,3,4 becomes a five digit binary number, recall digital systems)
So, if text is labeled with [0,2,3], it becomes [10110]
Why do we do this? Because the model expects this specific labeling. It expects float 0 or float 1
can be called one hot labeling btw

#What input will go into this function? a list of labels
{
  [0,1], [0], [4], [0,1,2,3], [2,3]
}

#What output will we get? a numpy array with a shape (num_samples, num_labels). In the example above we have 5 samples
#Output will be like this:
{
  [1,1,0,0,0],
  [1,0,0,0,0],
  [0,0,0,0,1],
  [1,1,1,1,0],
  [0,0,1,1,0]
}
'''
def binarize_labels(labels_list, num_labels=NUM_LABELS):
    #float32 here to match PyTorch's default tensor type. This helps speed and memory
    Y = np.zeros((len(labels_list), num_labels), dtype=np.float32)
    for i, labs in enumerate(labels_list):
        for j in labs:
            if 0 <= j < num_labels:
                Y[i, j] = 1.0
    return Y




#Tokenizing our batch using our preloaded tokenizer (Recall first cell)
'''
input is a dictionary with at least two keys that have different values. Our example will use text and labels keys and those keys will point to strings for the text key and lists of labels for the labels key.
{
  'text':
  [
    "We collect your data to improve services.",
    "You may opt out of personalized ads."
  ],

  'labels':
  [
    [0, 2],    #Data Collection, Data Sharing
    [3]        #User Control
  ]
}

#output is a dictionary returned by the tokenizer containing:
#input_ids (the token ids for each sentence's words), attention_mask (1 for real 0 for not), possibly token_type_ids (depending on model), we also attach labels (the one hot encoded labels)

#Output will look like this:
{
  'input_ids':
  [
    [101, 2057, 2434, 2115, 2951, 102],
    [101, 2017, 2453, 2041, 1997, 102]
  ],

  'attention_mask':
  [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1]
  ],

  'labels': [
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
  ]
}

#Input IDs are the IDs of the word in the model's vocabulary.

TOPUTINMODELEXPL
#What are embeddings? Basically a way to understand words using numbers. Each word ID is mapped to an embedding vector (of Length 768 for PrivBERT)
#These vectors have a bunch of different numbers [0.12,-0.54,0.91,.........,0.03] which represent the word. Words like coffee and espresso will have similar embeddings in terms of numbers.
#This means that the model starts to understand things more clearly.
#The embedding layer is the first layer in PrivBERT. TYhen they flow into Transformer layers
'''
def tokenize_batch(batch):

    '''
    VERY IMPORTANT
    Big comment regarding the next line:
    #Use tokenizer to go through the text part of our batch

    #Truncation true to make sure that sequence longer than max length are cut instead of error
    #Why keep truncation true? Handling runtime error on long policy statements. Tokenizer can raise errors if inputs exceed model's max
    #This will result in a trade off where we lose trailing information (textual info towards the end of sentences)

    #False padding because its just not needed. Nothing will happen if we do, nothing happens if we don't
    #This is the case because we use a DataCollator later which automatically pads for us. If we didnt use it we might need to make padding true

    #What is max length here? First we need to understand what a tokenizer does.
    #Tokenizer basically turns text into tokens. Each word takes a certain amount of tokens, usually 1-2 tokens.
    #Max length determines the maximum number of tokens we use. If we use 256 that means we have around 170~210 words of english language.
    #This gives a tradeoff, we might lose more tail information, which is why prof wanted to make sure we understand this.
    #We should ask her for what to do with max length.
    '''


    enc = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)


    #returns enc with our multi hot label matrix appended
    enc['labels'] = binarize_labels(batch['labels']).astype('float32').tolist()  # ensure float32
    return enc

# %%
#NEXT BLOCK Train Test Validation Split

'''#Dataset class from the datasets library provided by Hugging Face
#Dataset is a better pandas basically, more fitting for ML and NLP tasks'''
from datasets import Dataset

# Convert all labels to binary matrix form for stratification
#In detail, take the dataframe and split only the labels part, make it a list of labels, then one hot encode that.
Y_all = binarize_labels(df['labels'].tolist(), NUM_LABELS)

#Function to split into 80-10-10 Train test validation.
def multilabel_split(df, Y, val_size=0.1, test_size=0.1, seed=42):

    #train_test_split explanation
    '''
    #The function is used to split data into training and testing.
    #The way it works is that it takes some parameters as inputs (X,y, test_size, and random state), which were explained earlier
    #It returns X values for Train and Test, and same for Y.
    '''
    from sklearn.model_selection import train_test_split

    # Represent each label set as a string so its ready for stratification
    labelset = df['labels'].apply(lambda L: ';'.join(map(str, L)) if L else 'none')


    '''#Had to add these two lines to fix error.
    #vc counts labels. So it will have [count0,count1,count2,...countN], and that is the number of appearances of each label combination occurs [0,2],[0], [0,1,2,3], etc
    #next line handles case of a combination being <2, because it causes stratification to bug'''
    vc = labelset.value_counts()
    labelset = labelset.where(labelset.map(vc) >= 2, '__rare__')

    #First split: train+val vs test (seperate the 10% for test and ensure distribution)
    #We are inputting one dataset (the set of indexes), giving test_size, seed, and labelset for stratify
    temp_idx, test_idx = train_test_split(
        np.arange(len(df)), #an array of sample IDs. It will be [0,1,2,3,4,...,n]. We do this to later reference the index for efficiency and simplicity.
        test_size=test_size,
        random_state=seed,
        stratify=labelset #This is where we ensure distribution
        )

    #Second split: train vs validation (Seperate the rest and ensure distribution)
    '''
    #labelset is the full set of labels from our dataset.
    #we take the 90% that we got after we filtered out for testing (temp_idx) and were left with labelset_temp
    #now we use train_test_split again to split those such that we have a final 80-10-10 split
    '''
    labelset_temp = labelset.iloc[temp_idx]
    #split into train_idx and val_idx. So we have test_idx, train_idx, and val_idx
    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=val_size / (1 - test_size), #same process but take into account that we already seperated test
        random_state=seed,
        stratify=labelset_temp #This is where we ensure distribution
        )


    #Return stratified and cleaned dataframes
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True)
        )


'''#Perform the 80/10/10 split
#multilabel_split takes x,y, sizes of validation and testing, and a seed for reproducibility
#returns 3 sets of values (train, validation, and test)'''
train_df, val_df, test_df = multilabel_split(df, Y_all, val_size=0.1, test_size=0.1, seed=42)

'''
#Shape of our training set is: (number_of_samples_in_train_set, 2) because we have a certain amount of samples and only two columns, text and labels.
#this is the shape of our datasets NOW, not the shape that will go into the model. We are still going to downsample and then binarize and tokenize.
#After one hot encoding and tokenizing, we will have 2 tensors. Label Tensor and Text Tensor.
#Label Tensor shape: (N,5), N rows and 5 columns because we have 5 labels.
#Text Tensor shape: (N,max_length) where max length is the same as the one we input in Tokenizer. This will be the fixed length with padding and truncation.'''
print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

# %%
#NEXT BLOCK Creating downsampled test set.

import numpy as np
import pandas as pd
from datasets import Dataset

#'Other' class ID in id2label
OTHER_ID = 4

#Returns True if labels contain only 'Other'
def is_other_only(lbls):
    s = sorted(set(lbls))
    return (len(s) == 1) and (s[0] == OTHER_ID)

#distribution of labels
print("DISTRIBUTION OF TRAINING SET LABELS before downsampling")
print(train_df['labels'].explode().value_counts())



#Downsample rows where labels = [4] to a fraction (30% by default) of their original count. Keeps everything else same.
def downsample_other_only(df: pd.DataFrame, keep_frac=0.3, seed=42) -> pd.DataFrame:

    #Uses seed to ensure reproducibility
    rng = np.random.default_rng(seed)

    #Only the rows containing other
    mask_other_only = df['labels'].apply(is_other_only)

    #Turn it into a dataframe
    other_only_df = df[mask_other_only]

    #The rest is turned into another dataframe. '~' means inverted
    rest_df = df[~mask_other_only]


    #Builds a smaller dataframe called kept_other_only (this is the downsampled dataframe now)
    k = max(1, int(len(other_only_df) * float(keep_frac)))
    keep_idx = rng.choice(other_only_df.index.to_numpy(), size=k, replace=False) #keep_idx is an array of indices to be kept.
    kept_other_only = other_only_df.loc[keep_idx] #kept_other_only is the array of samples that are kept


    '''#Merge kept other only with rest and store in balanced
    #concatenate the downsampled others, and the rest
    #reshuffle them
    #reset indices'''
    balanced = pd.concat([rest_df, kept_other_only], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


    #Print summary and return balanced
    print(f"Test Other-only: {len(other_only_df)} → {k} (keep_frac={keep_frac})")
    print(f"New display_test size: {len(df)} → {len(balanced)}")
    return balanced

# Keep an untouched test copy for real evaluation
official_test_df = test_df.copy()

# Create a downsampled test for showing predictions (not for metrics)
display_test_df = downsample_other_only(test_df, keep_frac=0.2, seed=42)

#Downsampling training data
train_df = downsample_other_only(train_df, keep_frac=0.2, seed = 42)

#Downsampling validation data
val_df = downsample_other_only(val_df, keep_frac=0.2, seed=42)


# Convert DataFrames into tokenized Hugging Face Datasets
def make_dataset(dframe):

    #Cols? Unneeded columns. Will be dropped after tokenization
    cols = ['text', 'labels', '__index_level_0__'] if '__index_level_0__' in dframe.columns else ['text', 'labels']
    return (

        #Turns Dataframe into Hugging Face Dataset
        Dataset.from_pandas(dframe.reset_index(drop=True))

        #Applies tokenizer function that we coded to every batch of rows
        .map(tokenize_batch, batched=True, remove_columns=cols)

        #Ensures that output tensors can work with PyTorch
        .with_format(type='torch')
    )


#Build all splits
train_ds = make_dataset(train_df)
val_ds   = make_dataset(val_df)
test_ds = make_dataset(test_df)              # untouched, used for evaluation
display_test_ds = make_dataset(display_test_df)  # optional, for showcasing results


print(train_df.head)

print(f'dataset that will be used for training has {len(train_ds)} values' )
print(f'dataset that will be used for validation has {len(val_ds)} values' )
print(f'dataset that will be used for testing later has {len(display_test_ds)} values' )

#distribution of labels
print("DISTRIBUTION OF TRAINING SET LABELS after downsampling")
print(train_df['labels'].explode().value_counts())

# %%
#NEXT BLOCK Building Model

#Loading a PRETRAINED transformer model (PrivBERT in this case) and configuring it for multi-label classification.
'''
#AutoModelForBlablabla will automatically pick the right architecture (BERT, RoBERTa, DistilBERT, etc) based on the model name we gave it.
#.from_pretrained means: pick from the pretrained ones. We will handle fine tuning
#model.to(DEVICE) means move the model weights into GPU if possible, else move it into CPU (defined and explained earlier)

#Why does the output say roBERTa? VERY IMPORTANT
#Basically privBERT is an implementation of RoBERTa that is trained on priv policy data. The thing is, the authors that made mukund/privBERT had the checkpoint with a config.json
#this json sets model_type: "roberta". Their implementation's model is not EXPLICITLY named as "PrivBERT" but its basically PrivBERT.
'''
'''
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, #privbert from hugging face
    num_labels=NUM_LABELS, #5 labels (0-4)
    problem_type='multi_label_classification',
    id2label=id2label,
    label2id = {v: k for k, v in id2label.items()},
    use_safetensors=True
)'''

#model = AutoModelForSequenceClassification.from_pretrained("privbert_final")
model.to(DEVICE)

# %%
#NEXT BLOCK Trainer
import torch #Pytorch!
import torch.nn.functional as F #Gives you functional forms of losses/activations (used for BCE)
from transformers import Trainer #Hugging Face trainer

#Transformer architecture:
#Data gets embedded and tokenized. Both concepts explained above
#No need for positional embedding in our case
#

#Version-proof: accept extra kwargs like num_items_in_batch

#What does BCE mean? Stands for Binary Cross Entropy.
#
class BCETrainer(Trainer): #Custom trainer that inherits from Hugging Face's Trainer

    #Function to override the loss computing function in Hugging Face trainer
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        '''
        #IMPORTANT
        #Remove the input's labels and put it in a variable called 'labels'
        #to(model.device) to ensure that no CPU/GPU complications occur (labels moved to the same device)
        #needs to be float cuz binary cross-entropy in a multi-label setting requires float
        #Labels is a tensor here, she might ask what a tensor is. A tensor is a container for numbers.
        #Can be 0D (scalar, e.g 1), can be 1D (List, [0,1,2]), can be 2D (a matrix of numbers), and so on.
        #Labels is a 2D Tensor [batch_size, num_labels]
        #Why do we do it?
        '''
        labels = inputs.pop("labels").to(model.device).float()

        #puts all remaining inputs (e.g., input_ids, attention_mask, possibly token_type_ids) into model
        #outputs is now a new model which does NOT use the built in loss because we want explicit control over that
        outputs = model(**inputs)

        #IMPORTANT
        '''
        #What are logits? She might ask this. Remember AI Course? Or ML even?
        #Logits are scores for every label basically. So for our 5 labels, we could have [-2.3, 4.1, 0.8, -0.4, 1.5].
        #Each number represents how likely it is that the label is correct. If it is a large negative number? Very wrong. Large Positive? Very Right. Close to 0? Very Unsure
        #If we turn these into sigmoids (or probabilities between 0 & 1) it will look like this [0.09, 0.98, 0.69, 0.40, 0.82]
        '''
        #Next line extracts the raw, unnormalized scores (AKA Logits)
        logits = outputs.logits


        '''
        #What is BCE (Binary Cross Entropy)? Basically a method of training models
        #Lets model predict. If the logits are high in value but not matching the true values -> Penalize. Penalize more if its very confident and wrong.
        #Do this a bunch of times then take the average penalty.
        #FYI 'loss' is basically the penalty. If loss is high, prediction is very inaccurate. If it is very low, prediction was very accurate.
        '''
        #Each label is treated independently, so we find the correct loss for multilabel classification. Each label has its own average loss.
        #This next line is done to explicitly avoid sigmoids and use logits instead. Why avoid logits? to ensure that the function handles conversions and we avoid errors
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss

'''
#Rebuild trainer (same kwargs you used before)
#What are kwargs? Basically extra information. Literally stands for Keyword Arguments
#If you have a function 'bilal' that takes 'salman' and 'ahmed', adding kwargs to parameters basically means that any extra parameters will be in kwargs
#kwargs functions like a dictionary, in use, you have to say x = y (or student = saeed)
#if definition is bilal(salman, ahmed, **kwargs), but call is bilal(salman,ahmed, student='saeed', prof='reham') -> our parameters are now [salman, bilal, student='saeed', prof='reham']
#in our case its like a safety net, cuz if our function takes too many parameters, it will probably crash, so we use kwargs.
'''
from transformers import TrainingArguments
args = TrainingArguments(
    output_dir="privbert-finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    logging_dir="./logs",      # old-style logging
    save_steps=500,            # save every 500 steps
)

#These are our kwargs that we will put into our BCETrainer
trainer_kwargs = dict(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer = tokenizer
)


#Look at BCETrainer, if it has 'processing_class, set a new variable ('processing_class') in trainer_kwargs to tokenizer, else set a new variable ('tokenizer') to tokenizer.
#This is done just to add some things that BCETrainer might expect, we want to add as many kwargs to ensure that it works right.
if "processing_class" in BCETrainer.__init__.__code__.co_varnames:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

#set variable trainer to a new instance of BCETrainer filled with our trainer_kwargs
trainer = BCETrainer(**trainer_kwargs)
print("Using BCETrainer (kwargs-compatible)")

# %%
#NEXT BLOCK Train and Evaluate (Took 27 mins to run)
'''
#Epoch loop begins: Trainer goes through training dataset multiple times, each full pass is a called an epoch
#Mini Batches are created: Dataset is divided into batches (8,16, or 32 at a time). DataCollator automatically pads them and cleans
#Forward Pass: Each batch of formatted and tokenized inputs gets fed into model, then model makes logits
#Loss Calculation: BCE is done to calculate loss
#Backpropagation: PyTorch does its work and calculates gradients (how much each weight in the model contributed to the loss).
#The gradients are used to slightly update the model to be more accurate
#Logging and Checkpoints: every few epochs, progress is logged (learning rate, evaluation results, etc) and checkpoints are saved while logging.
'''


#This next command is where we fine tune privBERT. Bunch of things happen in the order shown above ^^
#Command will return a "TrainOutput" which has the final loss and some metrics (like total epochs, runtime, samples per second, etc)
#train_result = trainer.train()

#switches model to evaluation mode and uses the validation set to produce prediction logits which are compared to the real values and metrics are produced
#eval_metrics = trainer.evaluate()

#Show the metrics from validation set
#eval_metrics
#trainer.save_model("privbert_final")
#tokenizer.save_pretrained("privbert_final")

# %%

import matplotlib.pyplot as plt
#NEXT BLOCK plotting learning curve
# Trainer keeps a log of training/eval in trainer.state.log_history
logs = trainer.state.log_history

train_epochs, train_losses = [], []
val_epochs,   val_losses   = [], []

for log in logs:
    # training loss entries
    if "loss" in log and "epoch" in log:
        train_epochs.append(log["epoch"])
        train_losses.append(log["loss"])
    # validation loss entries (only if your Trainer logged eval during training)
    if "eval_loss" in log and "epoch" in log:
        val_epochs.append(log["epoch"])
        val_losses.append(log["eval_loss"])

plt.figure(figsize=(6,4))

plt.plot(train_epochs, train_losses, marker='o', label='Training loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=200)  # so you can use it in the report
plt.show()

# %%
'''
#Finding the best threshold of accepting true sigmoid values. Using f1 macro metric to judge our model
#Why f1 macro? Remember how f1 works. It penalizes when either precision or recall is bad.
#Precision: Of everything I predicted as positive, how many were actually positive?
#Recall: Of all the actual positives, how many did I find?
#Macro F1 is used because its better for multilabel problems. It basically computes each F1 alone then averages everything.
#This ensures that the metric is based on every label being counted fairly, even if there is a very dominant label (Others).
#For example, if our F1s were as follows [0.70, 0.68, 0.60, 0.72, 0.95], then each would be given the same weight when calculating average.
#This ensures that having a dominant label wont really matter
'''
#NEXT BLOCK metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

#Takes a validation set and returns probs and y_true. Probs is sigmoid values for the set. y_true is the actual value.
#Output looks like preds = [0.97,0.2,0.03,0.91,0.02] & y_true = [1,0,0,1,0].

'''
    #Disables gradient tracking and does forward passing while unpacking dict keys in our model to find logits. Stored in 'logits'
    #Why do we disable gradient tracking? Usually when model(**toks).logits is called, PyTorch will find gradients and that will be resource costly (time and memory)
    #Just disable for quicker code since were only doing a forward pass. No need for gradients
    #What we did here is that we took the toks defined earlier and did a forward pass (gave toks as input and went through the whole model).
    #But before that, we did **toks which unpacks the dict so that it can be inputted into the model straightaway.
    #We did this because we want to get pure output scores and sigmoid them. This will help us in tuning our thresholds
    
    #Applies sigmoid to the logits. Turns them into NumPy array for scikit-learn usage (sklearn accepts NumPy)
    #We put it in CPU because you cant convert GPU tensors directly to NumPy arrays, has to be on the CPU
'''

def get_val_logits_labels(val_texts, val_labels, batch_size = 32):
    '''
    #Tokenizing all validation text at once
    #cutting any sequence longer than 256 tokens
    #padding anything less than the max length of text
    #Return value is PyTorch tensors not python lists
    #As explained earlier, tokenizer will return a dict of tensors. input_ids(N,256) - attention_mask(N,256)'''
    all_probs = []
    for start in range(0, len(val_texts),batch_size):
        batch_texts = val_texts[start:start + batch_size]
        toks = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**toks).logits #Output is a PyTorch tensor of shape (num_of_samples, num_of_labels)
            batch_probs = torch.sigmoid(logits).cpu().numpy() #Output is a NumPy array of same shape but its sigmoided (so probabilities now)
            all_probs.append(batch_probs)
    
    probs = np.concatenate(all_probs, axis=0) 

    #Binarizing y_true (one hot encoding). Output will be (num_of_samples, num_of_labels)
    y_true = binarize_labels(val_labels, NUM_LABELS)
    return probs, y_true




#Runs the function on our validation set
val_probs, val_true = get_val_logits_labels(val_df['text'].tolist(), val_df['labels'].tolist())


# %%
#NEXT BLOCK Per Label Threshold function
'''
#This function goes over the acceptance thresholds for each label. This is better than having one global threshold because it makes sure the model can make a different decision for each label
#How does it work? Loops through validation set's predictions and tries different thresholds. Finds the one with the highest f1 score for EACH label and sets that
#Function will return 3 values. A vector of thresholds [0.2,0.4,0.75,0.9,0.1] for each label. Same for F1s. And a single number for the average of the best f1s
'''

#Will take probs, y_true, and grid as inputs. probs is the sigmoids calculated earlier. y_true is the actual true values, and grid: array of possible threshold values
#Grid looks like: [0.05,0.10,0.15,...0.95]
def tune_thresholds_per_label(probs, y_true, grid=np.linspace(0.05, 0.95, 19)):
    num_labels = probs.shape[1] #Number of labels
    thresholds = np.zeros(num_labels, dtype=float) #Create an np array of 0s for thresholds
    f1_per_label = np.zeros(num_labels, dtype=float) #Same for f1 scores

    for n in range(num_labels): #Loop through labels
        best_t, best_f1 = 0.5, -1.0 #placeholders such that any candidate will beat it
        for t in grid: #Loop through threshold candidates
            preds_L = (probs[:, n] >= t).astype(int) #preds calculated using one of those candidates
            f1L = f1_score(y_true[:, n], preds_L, average='binary', zero_division=0) #F1 scores for that candidate
            if f1L > best_f1: #if it is good, set
                best_f1, best_t = f1L, t
        thresholds[n] = best_t #Set the threshold
        f1_per_label[n] = best_f1 #Set the f1

    macro_f1_at_best = f1_per_label.mean() #Get the macro f1 by averaging the f1s
    return thresholds, f1_per_label, macro_f1_at_best


#Run per-label threshold tuning
BEST_THRESHOLDS, F1S_PER_LABEL, MACRO_F1 = tune_thresholds_per_label(val_probs, val_true)

#Printing
print("\nBest thresholds per label:")
for i, label_name in id2label.items():
    print(f"  {label_name:15s} → threshold={BEST_THRESHOLDS[i]:.2f}, F1={F1S_PER_LABEL[i]:.3f}")

print(f"\nMacro F1 (per-label tuned): {MACRO_F1:.3f}")

#Compute Macro Precision, Recall, Accuracy using tuned thresholds

#Convert probabilities to binary predictions using best thresholds
y_pred = (val_probs >= BEST_THRESHOLDS[None, :]).astype(int)

#Macro Precision
macro_precision = precision_score(val_true, y_pred, average='macro', zero_division=0)

#Macro Recall
macro_recall = recall_score(val_true, y_pred, average='macro', zero_division=0)

#Macro Accuracy
macro_accuracy = accuracy_score(val_true, y_pred)

print("\nAdditional Metrics")
print(f"Macro Precision → {macro_precision:.3f}")
print(f"Macro Recall    → {macro_recall:.3f}")
print(f"Subset Accuracy → {macro_accuracy:.3f}")

# %%
#NEXT BLOCK Output Cell
#Process? First we take the display test dataset

from sklearn.metrics import multilabel_confusion_matrix #Produces a 2x2 confusion matrix for each label.
import matplotlib.pyplot as plt #plotting library from python


#Function to predict any text. Takes one inputs, texts (a list of strings)
def classify(texts: List[str]):
    #Tokenize your texts into inputs for model and put them in GPU
    toks = tokenizer(texts, truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(DEVICE)

    #Forward pass
    with torch.no_grad():
        logits = model(**toks).logits
        probs = torch.sigmoid(logits).cpu().numpy() #shape (num_of_samples, num_of_labels)

    #Binarize the predictions (1 or 0) based on the thresholds. If less, 0 if equal or more 1
    binarized = (probs >= BEST_THRESHOLDS[None, :]).astype(int)
    #Guarantee at least one label per text
    for i in range(binarized.shape[0]):
        if binarized[i].sum() == 0: #If no labels, find the highest probability and set the according label to 1, its most likely others
            j = probs[i].argmax()
            binarized[i, j] = 1

    #Convert predictions into readable output
    results = []
    for t, row in zip(texts, binarized):
        ids = np.where(row == 1)[0].tolist()
        labels = [id2label[i] for i in ids]
        results.append((t, {'ids': ids, 'labels': labels}))
    return results


def predict_on_dataframe(dframe, batch_size=64):
    texts = dframe['text'].astype(str).tolist() #Take the dataframe's text col and convert it to a list of strings (so we can input it in our function)
    results = []

    for start in range(0, len(texts), batch_size): #for the texts, go through batches and add onto results the predicted labels for the texts
        chunk = texts[start:start+batch_size]
        results.extend(classify(chunk))

    return results

#Printing the results
def printResults(tested):
    for sent, info in tested[:150]:
        print(f"{sent}\n→ {info}\n")

test_preds = predict_on_dataframe(display_test_df)
printResults(test_preds)

y_true = binarize_labels(display_test_df['labels'].tolist(), NUM_LABELS)
y_pred = np.zeros_like(y_true, dtype=int)
for i, (_, info) in enumerate(test_preds):
    for lid in info['ids']:
        y_pred[i, lid] = 1

cms = multilabel_confusion_matrix(y_true, y_pred)

# %%
#Confusion Matrix
for i, label_name in id2label.items():
    tn, fp, fn, tp = cms[i].ravel()
    mat = np.array([[tn, fp],[fn, tp]])
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(mat, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0','Pred 1']); ax.set_yticklabels(['True 0','True 1'])
    for (r,c), v in np.ndenumerate(mat):
        ax.text(c, r, int(v), ha='center', va='center')
    ax.set_title(f'Confusion Matrix – {label_name}')
    plt.tight_layout(); plt.show(block = False)

'''
import glob
def txt2List(path):
    txtPath = glob.glob(path)
    txt = []
    for path in txtPath:
        with open(path, "r", encoding="utf-8") as f:
            content=f.read().strip()
            txt.append(content)
    return txt
'''

# %%
import glob
import re
def txt_to_sentences(pattern):
    """
    Reads all txt files matching 'pattern' and returns a list of sentences.
    Each sentence is a separate string ready to be classified.
    """
    txt_paths = glob.glob(pattern)
    all_sentences = []

    for path in txt_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Simple sentence split: split on ., ?, ! followed by whitespace/newline
        sentences = re.split(r'(?<=[.!?])\s+', content)
        # Clean and drop empty ones
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sentences.extend(sentences)
    return all_sentences

def printOutput(num):
    print(f"Output for {num}.txt")
    txt = txt_to_sentences(f"{num}.txt")
    if not txt:
        print("ERROR NO SENTENCES FOUND IN THIS FILE. SKIPPED")
        return
    txtpred = classify(txt)
    printResults(txtpred)

for x in range(1,11):
    printOutput(x)
# %%

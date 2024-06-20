import os
import pandas as pd
from datasets import load_dataset

labels_dict = {
    "trec": ['ABBR','ENTY','DESC','HUM','LOC','NUM'],
    "ag_news": ["World", "Sports", "Business", "Sci/Tech"]
}

def save_dataset(dataset, split, filename):
    data = dataset[split].to_pandas()
    data = data[['text', 'label' if 'label' in data.columns else 'coarse_label']].rename(columns={'text': 'question', 'label': 'label', 'coarse_label': 'label'})
    print(filename, data.shape)
    data.to_csv(filename, index=False)

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load AG News dataset
    ag_news_dataset = load_dataset('fancyzhx/ag_news')
    save_dataset(ag_news_dataset, 'train', './data/ag_news_train.csv')
    save_dataset(ag_news_dataset, 'test', './data/ag_news_test.csv')

    # Load TREC dataset
    trec_dataset = load_dataset('CogComp/trec', trust_remote_code=True)
    save_dataset(trec_dataset, 'train', './data/trec_train.csv')
    save_dataset(trec_dataset, 'test', './data/trec_test.csv')

if __name__ == '__main__':
    main()

TRAIN_DATA_PATH = "data/train_how-to-mua-nha.csv"
TEST_DATA_PATH  = "data/test_how-to-mua-nha.csv"

import pandas as pd 
import logging
import csv
from sklearn.model_selection import train_test_split

# logging.basicConfig(filename="log/preprocess.log", filemode="w", level=logging.DEBUG, format="%(level)s\t%(message)s")
logging.basicConfig(filename='log/preprocess.log',filemode='w', format='%(levelname)s\t%(message)s', level=logging.DEBUG)

def divide_train_test(file_dataset):
    with open(file_dataset, 'r') as f:
        rows = csv.reader(f)
        list_data = list(rows)
    train_set, test_set = train_test_split(list_data, test_size = 0.2, random_state=40)
    
    with open("data/train_set.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(train_set)

    with open("data/test_set.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(test_set)
    

def onehotencode(column_data_pd):
    encode_obj = {
        'INLAND':[1,0,0,0,0],
        '<1H OCEAN': [0,1,0,0,0],
        'ISLAND': [0,0,1,0,0],
        'NEAR OCEAN': [0,0,0,1,0],
        'NEAR BAY': [0,0,0,0,1]
    }
    gan_bien_encode = [encode_obj[item] for item in column_data_pd]
    gan_bien_encode = pd.DataFrame(gan_bien_encode, columns= ['INLAND', '<1H OCEAN','ISLAND', 'NEAR OCEAN','NEAR BAY'])
    return gan_bien_encode

def main():
    #load data
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    logging.debug("Train data:\n%s", train_data)
    

# main()
divide_train_test("data/dataset.csv")
from feature_extractor import *
import os
import shutil
import argparse
import warnings 
import glob
import pickle
from sequentia.preprocessing import Compose, Custom, Standardize
from sequentia.classifiers import GMMHMM, HMMClassifier
from sklearn.model_selection import train_test_split,KFold
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Baseline framework')
parser.add_argument('--data_dir', default='', type=str, metavar='PATH',
                    help='path of the data directory')
parser.add_argument('--pet', default='', type=str,
                    help='path containing input audio file')
parser.add_argument('--folds', default=-1, type=int,
                    help='Number of folds for cross validation')

# Directories                   
cwd = os.getcwd()

def preprocess(data_dir, pet_type):

    print("---------- Preprocessing ---------")
    print("")
    print("=> Organizing data ...")
    data_dir = os.path.join(cwd, data_dir)
    isolated_dir = os.path.join(cwd,'data/isolated')
    if not os.path.exists(isolated_dir):
        os.makedirs(isolated_dir)
    for fpath in glob.iglob(f"{data_dir}/**/*{pet_type}.wav",recursive = True):
        if not os.path.exists(os.path.join(isolated_dir,fpath.split('/')[-1])):
            shutil.copy2(fpath,isolated_dir)
            shutil.copy2(fpath.split('.wav')[0]+'.csv',isolated_dir)
            
    piece_dir = os.path.join(cwd,'data/piece')
    if not os.path.exists(piece_dir):
        os.mkdir(piece_dir)
    for fpath in glob.iglob(f"{data_dir}/**/*.wav",recursive = True):
        if not os.path.exists(os.path.join(piece_dir,fpath.split('/')[-1])):
            if 'Piece' in fpath.split('/')[-1]:
                shutil.copy2(fpath,piece_dir)

    for fpath in glob.iglob(f"{data_dir}/**/*_tech_{pet_type}.csv",recursive = True):
        if not os.path.exists(os.path.join(piece_dir,fpath.split('/')[-1])):  
            shutil.copy2(fpath,piece_dir)

    print("=> Extracting pitch tracks ...")

    for fpath in glob.iglob("data/**/*.wav",recursive = True):
        pitch,fname = extractPitch(fpath, fs=44100, overwrite = False, vampdir='../')

def evaluate(pet_type, cvd=-1):

    print("---------- Evaluation ---------")
    print("")
    print(f"=> Extracting features for {pet_type} evaluation...")
    data_dict = {}

    for idx, fpath in enumerate(glob.iglob(f"data/**/*.wav",recursive = True)):
        print(f"Parsing ---------> {fpath.split('/')[-1]}")
        tokens, targets = extract_features(fpath, feature_type=pet_type)
        if tokens is None:
            continue
        data_dict[str(idx)] = {}
        data_dict[str(idx)]['path'] = fpath
        data_dict[str(idx)]['tokens'] = tokens
        data_dict[str(idx)]['targets'] = targets
        
    with open('data/feature_dict.pkl','wb') as fp:
        pickle.dump(data_dict,fp)


    X = []
    y = []
    for datafile in data_dict.values():
        for token in datafile['tokens']:
            X.append(token)
        for t in datafile['targets']:
            y.append(t)

    if cvd >= 2:
        n_splits = 5
        kf = KFold(n_splits=n_splits,shuffle=True)
        F1_score = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):

            print(f"=> Validating fold --------- {i}/{n_splits}")
            X_train = [X[ix] for ix in train_index]
            y_train = [y[ix] for ix in train_index]
            X_test = [X[ix] for ix in test_index]
            y_test = [y[ix] for ix in test_index]

            # Sort according to label
            pos_seq = [data[0] for data in list(zip(X_train,y_train)) if data[1]==1]
            neg_seq = [data[0] for data in list(zip(X_train,y_train)) if data[1]==0]

            train_set_by_label = [neg_seq,pos_seq]
            print(len(train_set_by_label[0]))
            print(len(train_set_by_label[1]))

            random_state = np.random.RandomState(1)
            hmms = []
            for idx, seq in enumerate(train_set_by_label):
                print(f"Training HMM {idx}...")
                hmm = GMMHMM(label=idx, n_states=3, n_components=2, topology='linear', random_state=random_state)
                hmm.set_random_initial()
                hmm.set_random_transitions()

                hmm.fit(seq)
                hmms.append(hmm)

            clf = HMMClassifier().fit(hmms)
            acc, cm = clf.evaluate(X_test, y_test, verbose=False, n_jobs=-1)
            pre = cm[0][0]/(cm[0][0] + cm[1][0]) 
            rec = cm[0][0]/(cm[0][0] + cm[0][1]) 

            F1_score.append(pre*rec/(pre+rec))
            sc = sum(F1_score)/n_splits

        print("")
        print(f"F1_score for the {pet_type} classification task: {(sc * 100):.2f}%")

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Sort according to label
        pos_seq = [data[0] for data in list(zip(X_train,y_train)) if data[1]==1]
        neg_seq = [data[0] for data in list(zip(X_train,y_train)) if data[1]==0]

        train_set_by_label = [neg_seq,pos_seq]

        random_state = np.random.RandomState(1)
        hmms = []
        for idx, seq in enumerate(train_set_by_label):
            print(f"=> Training HMM {idx}...")
            hmm = GMMHMM(label=idx, n_states=3, n_components=2, topology='linear', random_state=random_state)
            hmm.set_random_initial()
            hmm.set_random_transitions()

            hmm.fit(seq)
            hmms.append(hmm)
            
        clf = HMMClassifier().fit(hmms)
        acc, cm = clf.evaluate(X_test, y_test, verbose=False, n_jobs=-1)
        pre = cm[0][0]/(cm[0][0] + cm[1][0]) 
        rec = cm[0][0]/(cm[0][0] + cm[0][1]) 
        sc = pre*rec/(pre+rec)
         

        print("")
        print(f"F1_score for the {pet_type} classification task: {(sc * 100):.2f}%")



def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    pet = args.pet
    folds = args.folds

    if pet == 'Glissando' or pet=='Portamento':
    
        preprocess(data_dir, pet_type=pet)
        evaluate(pet_type=pet, cvd=folds)

    else:
        preprocess(data_dir, pet_type='Glissando')
        evaluate(pet_type='Glissando', cvd=folds)
        preprocess(data_dir, pet_type='Portamento')
        evaluate(pet_type='Portamento', cvd=folds)


if __name__ == '__main__':
    main()

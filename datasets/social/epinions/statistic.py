import pickle

def load_pkl(file):
    with open(file, 'rb') as fs:
        data = pickle.load(fs)
    return data

if __name__=="__main__":
    category = load_pkl('category.pkl')

    trn_mat = load_pkl('trn_mat.pkl')
    print(trn_mat)
    exit()
    
    tst_mat = load_pkl('tst_mat.pkl')
    print(tst_mat)
    exit()
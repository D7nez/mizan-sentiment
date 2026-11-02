from sklearn.model_selection import train_test_split
from .preprocess import preprocess

def prepare(frame,text_column='review_description',label_column='rating'):
    if text_column not in frame or label_column not in frame:raise ValueError('missing_dataset_columns')
    data=frame[[text_column,label_column]].dropna().copy();data.columns=['text','label']
    mapping={'1':'pos','0':'neu','-1':'neg','pos':'pos','neu':'neu','neg':'neg'}
    values=data['label'].astype(str).str.replace(r'\.0$','',regex=True)
    if not values.isin(mapping).all():raise ValueError('unknown_labels_expected_-1_0_1_or_neg_neu_pos')
    data['label']=values.map(mapping);data['clean']=data['text'].astype(str).map(preprocess)
    data=data[data['clean'].str.len()>0]
    if (data.groupby('clean')['label'].nunique()>1).any():raise ValueError('conflicting_labels_for_same_cleaned_text')
    data=data.drop_duplicates('clean').reset_index(drop=True)
    if set(data['label'])!={'neg','neu','pos'} or data['label'].value_counts().min()<7:raise ValueError('need_at_least_7_unique_texts_per_class')
    return data

def split(data,seed=42):
    train,rest=train_test_split(data,test_size=.30,stratify=data.label,random_state=seed)
    val,test=train_test_split(rest,test_size=.5,stratify=rest.label,random_state=seed)
    sets=[set(x.clean) for x in [train,val,test]]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise RuntimeError('split_overlap')
    return train,val,test

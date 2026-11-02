import json
from pathlib import Path
import pandas as pd,numpy as np,pytest
from mizan.preprocess import preprocess,encode
from mizan.data import prepare,split
from mizan.evaluation import metrics
from mizan.storage import Audit

ROOT=Path(__file__).resolve().parents[1]
def test_negation_preserved_and_empty_inputs():
    for term in ['لا','ما','لم','ليس','لن','مو']:assert term in preprocess(term+' ممتاز').split()
    assert preprocess('http://example.com @user 123 😊')==''

def test_padding_oov_and_truncation():
    token={'word_index':{'<OOV>':1,'جيد':2},'num_words':3,'lower':True,'filters':'','split':' ','oov_token':'<OOV>'}
    a,stats=encode(['جيد مجهول جيد جيد'],token,3)
    assert a.tolist()==[[2,1,2]] and stats[0]['truncated'] and stats[0]['oov_ratio']==.25

def test_clean_split_has_no_duplicates_or_test_fit():
    df=pd.read_csv(ROOT/'data/example_synthetic.csv');df=pd.concat([df,df.iloc[:5]])
    data=prepare(df);assert len(data)==36
    train,val,test=split(data)
    assert not set(train.clean)&set(test.clean) and not set(val.clean)&set(test.clean)
    assert len(train)+len(val)+len(test)==36

def test_conflicting_labels_rejected():
    df=pd.read_csv(ROOT/'data/example_synthetic.csv');extra=df.iloc[[0]].copy();extra['rating']=-1
    with pytest.raises(ValueError,match='conflicting'):prepare(pd.concat([df,extra]))

def test_metrics_include_all_classes():
    result=metrics(['neg','neu','pos'],np.eye(3))
    assert result['macro_f1']==1 and result['label_order']==['neg','neu','pos']

def test_audit_feedback_foreign_key_and_no_text_storage(tmp_path):
    audit=Audit(tmp_path/'audit.sqlite')
    audit.save('request','v1',[{'label':'pos','needs_review':False}],.1)
    audit.feedback('request',0,'neg');audit.feedback('request',0,'pos')
    with pytest.raises(ValueError):audit.feedback('request',1,'neg')
    with pytest.raises(ValueError):audit.feedback('unknown',0,'neg')
    assert audit.stats()['feedback_items']==1 and audit.stats()['predictions']==1
    with audit.connect() as c:assert 'text' not in [r[1] for r in c.execute('PRAGMA table_info(predictions)')]

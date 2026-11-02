import pytest
from fastapi.testclient import TestClient
from mizan import api
from mizan.predictor import InputError
class Fake:
    def __init__(self,*args):self.version='test-fixture';self.evaluation_status='fixture'
    def predict(self,texts):
        if texts==['123']:raise InputError('empty_after_preprocessing:index=0')
        return [{'label':'pos','model_score':.8,'scores':{'neg':.1,'neu':.1,'pos':.8},'needs_review':False,'warnings':[],'token_count':2,'truncated':False,'oov_ratio':0} for _ in texts]
@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setattr(api,'Predictor',Fake);monkeypatch.setenv('AUDIT_DB',str(tmp_path/'audit.sqlite'));monkeypatch.setenv('APP_API_KEY','fixture-only')
    with TestClient(api.app) as client:yield client
H={'X-API-Key':'fixture-only'}
def test_auth_validation_and_feedback(client):
    assert client.post('/predict',json={'texts':['جيد']}).status_code==401
    for texts in [[],[''],['x'*5001],['جيد']*101,['123']]:assert client.post('/predict',json={'texts':texts},headers=H).status_code==422
    response=client.post('/predict',json={'texts':['خدمة جيدة']},headers=H)
    assert response.status_code==200 and response.json()['model_version']=='test-fixture'
    rid=response.json()['request_id']
    assert client.post('/feedback',json={'request_id':rid,'item_index':0,'label':'neg'},headers=H).status_code==200
    assert client.get('/monitoring',headers=H).json()['requests']==1

def test_csv_and_ui(client):
    for csv in ['text\n','wrong\na\n','text\n""\n']:
        assert client.post('/predict-csv',json={'csv_content':csv},headers=H).status_code==422
    result=client.post('/predict-csv',json={'csv_content':'text\n"خدمة جيدة"\n'},headers=H)
    assert result.status_code==200 and len(result.json()['results'])==1
    assert client.get('/').status_code==200 and client.get('/ui/app.js').status_code==200

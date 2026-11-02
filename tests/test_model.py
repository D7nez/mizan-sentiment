import os
from pathlib import Path
import pytest,numpy as np
@pytest.mark.skipif(os.getenv('RUN_MODEL_TESTS')!='1',reason='Set RUN_MODEL_TESTS=1 with TensorFlow and local artifact')
def test_real_legacy_model_load_inference_and_batch_parity():
    from mizan.predictor import Predictor,InputError
    predictor=Predictor(Path(__file__).resolve().parents[1]/'artifacts/legacy')
    texts=['الخدمة ممتازة وسريعة الاستجابة','تجربة سيئة جدا ما أنصح فيه','المنتج عادي لا بأس فيه']
    batch=predictor.predict(texts)
    assert [r['label'] for r in batch]==['pos','neg','neu']
    for text,row in zip(texts,batch):
        np.testing.assert_allclose(list(row['scores'].values()),list(predictor.predict([text])[0]['scores'].values()),atol=1e-5)
    with pytest.raises(InputError):predictor.predict(['123'])

@pytest.mark.skipif(os.getenv('RUN_MODEL_TESTS')!='1',reason='Requires TensorFlow and local artifacts')
def test_real_api_csv_prediction_feedback_and_readiness(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from mizan.api import app
    monkeypatch.setenv('MODEL_DIR',str(Path(__file__).resolve().parents[1]/'artifacts/legacy'))
    monkeypatch.setenv('AUDIT_DB',str(tmp_path/'real.sqlite'));monkeypatch.setenv('APP_API_KEY','model-test-only')
    h={'X-API-Key':'model-test-only'}
    with TestClient(app) as client:
        assert client.get('/ready').json()['ready']
        result=client.post('/predict-csv',headers=h,json={'csv_content':'text\nالخدمة ممتازة وسريعة الاستجابة\n'}).json()
        assert result['results'][0]['label']=='pos'
        response=client.post('/feedback',headers=h,json={'request_id':result['request_id'],'item_index':0,'label':'pos'})
        assert response.status_code==200 and client.get('/monitoring',headers=h).json()['feedback_items']==1

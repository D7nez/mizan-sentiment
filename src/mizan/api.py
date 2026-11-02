import os,time,secrets,logging,csv,io
from uuid import uuid4,UUID
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,Depends,Header,HTTPException,Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .predictor import Predictor,InputError
from .storage import Audit

class PredictionRequest(BaseModel):texts:list[str]=Field(min_length=1,max_length=100)
class Feedback(BaseModel):
    request_id:UUID
    item_index:int=Field(ge=0,lt=100)
    label:Literal['neg','neu','pos']
def auth(x_api_key:str=Header(default='')):
    key=os.getenv('APP_API_KEY','')
    if key and not secrets.compare_digest(x_api_key,key):raise HTTPException(401,'Invalid API key')
@asynccontextmanager
async def lifespan(app):
    app.state.predictor=Predictor(os.getenv('MODEL_DIR','artifacts/legacy'),float(os.getenv('REVIEW_THRESHOLD','.6')))
    app.state.audit=Audit(os.getenv('AUDIT_DB','state/audit.sqlite'))
    yield
app=FastAPI(title='Mizan Arabic Sentiment API',version='2.0.0',lifespan=lifespan)
@app.get('/health')
def health():return {'status':'ok'}
@app.get('/ready')
def ready():return {'ready':True,'model_version':app.state.predictor.version,'evaluation_status':app.state.predictor.evaluation_status}
@app.post('/predict',dependencies=[Depends(auth)])
def predict(body:PredictionRequest):
    if any(not t.strip() or len(t)>5000 for t in body.texts):raise HTTPException(422,'Each text must contain 1–5000 characters')
    request_id=str(uuid4());started=time.monotonic()
    try:results=app.state.predictor.predict(body.texts)
    except InputError as exc:raise HTTPException(422,str(exc)) from None
    duration=time.monotonic()-started
    app.state.audit.save(request_id,app.state.predictor.version,results,duration)
    logging.getLogger('mizan').info('request_id=%s batch_size=%s duration_s=%.4f',request_id,len(results),duration)
    return {'request_id':request_id,'model_version':app.state.predictor.version,'evaluation_status':app.state.predictor.evaluation_status,'duration_s':duration,'results':results,'score_notice':'Softmax model scores are not calibrated probabilities of correctness.'}
@app.post('/feedback',dependencies=[Depends(auth)])
def feedback(body:Feedback):
    try:app.state.audit.feedback(str(body.request_id),body.item_index,body.label)
    except ValueError as exc:raise HTTPException(404,str(exc)) from None
    return {'saved':True,'note':'Feedback is stored; no automatic retraining.'}
@app.get('/monitoring',dependencies=[Depends(auth)])
def monitoring():return app.state.audit.stats()
STATIC=Path(__file__).resolve().parent/'static'
app.mount('/ui',StaticFiles(directory=STATIC),name='ui')
@app.get('/',include_in_schema=False)
def home():return FileResponse(STATIC/'index.html',headers={'Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"})

class CSVRequest(BaseModel):
    csv_content:str=Field(min_length=1,max_length=1000000)
    text_column:str=Field(default='text',min_length=1,max_length=100)
@app.post('/predict-csv',dependencies=[Depends(auth)])
def predict_csv(body:CSVRequest):
    reader=csv.DictReader(io.StringIO(body.csv_content.lstrip('\ufeff')))
    if not reader.fieldnames or body.text_column not in reader.fieldnames:raise HTTPException(422,'CSV text column not found')
    texts=[]
    for row in reader:
        if len(texts)>=100:raise HTTPException(422,'CSV batch limit is 100 rows; split the file')
        value=row.get(body.text_column)
        if value is None or not value.strip():raise HTTPException(422,'CSV contains an empty text row')
        texts.append(value)
    if not texts:raise HTTPException(422,'CSV contains no data rows')
    result=predict(PredictionRequest(texts=texts))
    return {**result,'input_texts':texts}

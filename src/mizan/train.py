"""Split first; fit vocabulary only on train. Dataset is supplied by the owner."""
import argparse,hashlib,json,os,time
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .data import prepare,split
from .evaluation import metrics,LABELS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True);parser.add_argument('--output',default='reports/training')
    parser.add_argument('--text-column',default='review_description');parser.add_argument('--label-column',default='rating')
    parser.add_argument('--models',choices=['baseline','both'],default='both');parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args();out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise SystemExit('Output directory must be empty; choose a new experiment path.')
    out.mkdir(parents=True,exist_ok=True)
    frame=prepare(pd.read_csv(args.data),args.text_column,args.label_column)
    train,val,test=split(frame,args.seed)
    metadata={'seed':args.seed,'data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),'synthetic':args.synthetic,
        'split_sizes':dict(zip(['train','validation','test'],map(len,[train,val,test]))),'unique_clean_texts':len(frame),
        'split_policy':'normalized_unique_text_70_15_15_stratified','test_used_for_selection':False}
    # Hashes document partition membership without exporting user review text.
    partitions={k:[hashlib.sha256(t.encode()).hexdigest() for t in part.clean] for k,part in zip(['train','validation','test'],[train,val,test])}
    (out/'split_hashes.json').write_text(json.dumps(partitions,indent=2));(out/'dataset_manifest.json').write_text(json.dumps(metadata,indent=2))
    candidates=[]
    baseline=Pipeline([('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,max_features=100000)),('classifier',LogisticRegression(class_weight='balanced',max_iter=1000,random_state=args.seed))])
    started=time.monotonic();baseline.fit(train.clean,train.label)
    import joblib
    joblib.dump(baseline,out/'tfidf_logistic.joblib')
    def aligned(data):return baseline.predict_proba(data)[:,[list(baseline.classes_).index(label) for label in LABELS]]
    candidates.append(('tfidf_logistic',metrics(val.label,aligned(val.clean)),metrics(test.label,aligned(test.clean)),time.monotonic()-started))
    if args.models=='both':
        import tensorflow as tf
        from sklearn.utils.class_weight import compute_class_weight
        from .layers import build_model
        from .preprocess import encode,stopwords
        tf.keras.utils.set_random_seed(args.seed)
        tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(1)
        tok=tf.keras.preprocessing.text.Tokenizer(num_words=60000,oov_token='<OOV>');tok.fit_on_texts(train.clean)
        token={k:getattr(tok,k) for k in ['word_index','num_words','oov_token','filters','lower','split','char_level']}
        arrays=[encode(part.clean.tolist(),token)[0] for part in [train,val,test]]
        targets=[np.array([LABELS.index(label) for label in part.label]) for part in [train,val,test]]
        model=build_model(vocab=min(60000,len(tok.word_index)+1))
        model.compile(optimizer=tf.keras.optimizers.Adam(.0005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        weights=compute_class_weight('balanced',classes=np.arange(3),y=targets[0]);started=time.monotonic()
        model.fit(arrays[0],targets[0],validation_data=(arrays[1],targets[1]),epochs=args.epochs,batch_size=64,class_weight=dict(enumerate(weights)),callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.3,min_lr=1e-6)],verbose=2)
        candidates.append(('bilstm_attention',metrics(val.label,np.asarray(model(arrays[1],training=False))),metrics(test.label,np.asarray(model(arrays[2],training=False))),time.monotonic()-started))
        folder=out/'bilstm_attention';folder.mkdir();model.save(folder/'model.keras')
        (folder/'tokenizer.json').write_text(json.dumps(token,ensure_ascii=False));(folder/'labels.json').write_text(json.dumps(LABELS));(folder/'stopwords.json').write_text(json.dumps(sorted(stopwords()),ensure_ascii=False))
        from .artifacts import manifest
        manifest(folder,'retrained-'+str(args.seed),'clean_split_'+('synthetic_smoke' if args.synthetic else 'evaluated'))
    import mlflow
    Path('state').mkdir(exist_ok=True)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','sqlite:///state/mlflow.db'));mlflow.set_experiment('mizan-sentiment')
    records=[]
    for name,valid,heldout,elapsed in candidates:
        report={'model':name,'validation':valid,'test':heldout,'training_duration_s':elapsed,**metadata}
        path=out/(name+'.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({'seed':args.seed,'synthetic':args.synthetic,'data_sha256':metadata['data_sha256'],'split_policy':metadata['split_policy']})
            mlflow.log_metrics({'validation_macro_f1':valid['macro_f1'],'test_macro_f1':heldout['macro_f1'],'test_accuracy':heldout['accuracy'],'training_duration_s':elapsed})
            mlflow.log_artifacts(str(out));report['mlflow_run_id']=run.info.run_id
        records.append(report)
    selection=max(records,key=lambda r:r['validation']['macro_f1'])['model']
    (out/'comparison.json').write_text(json.dumps({'selected_by_validation':selection,'candidates':records,'note':'Synthetic runs are pipeline checks, not real-world accuracy. Legacy metrics are not comparable.'},ensure_ascii=False,indent=2))
    print('Saved reports:',out,'; selected on validation:',selection)
if __name__=='__main__':main()

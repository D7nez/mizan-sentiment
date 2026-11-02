import hashlib,json,threading
from pathlib import Path
import numpy as np
from .preprocess import preprocess,encode

class InputError(ValueError):pass
class Predictor:
    def __init__(self,directory,threshold=.6):
        import tensorflow as tf
        from .layers import SelfAttention
        self.path=Path(directory);self.threshold=threshold;self.lock=threading.Lock()
        manifest=json.loads((self.path/'manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            if hashlib.sha256((self.path/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('artifact_checksum_mismatch')
        self.model=tf.keras.models.load_model(self.path/'model.keras',custom_objects={'SelfAttention':SelfAttention},compile=False,safe_mode=True)
        self.tokenizer=json.loads((self.path/'tokenizer.json').read_text());self.labels=json.loads((self.path/'labels.json').read_text())
        self.length=int(self.model.input_shape[-1]);self.version=manifest['model_version'];self.evaluation_status=manifest['evaluation_status']
        self.words=set(json.loads((self.path/'stopwords.json').read_text()))
        if self.model.output_shape[-1]!=len(self.labels) or set(self.labels)!={'neg','neu','pos'}:raise RuntimeError('label_contract_mismatch')
        with self.lock:self.model(np.zeros((1,self.length),dtype='int32'),training=False)
    def predict(self,texts):
        cleaned=[preprocess(t,self.words) for t in texts]
        for i,text in enumerate(cleaned):
            if not text:raise InputError('empty_after_preprocessing:index='+str(i))
        array,stats=encode(cleaned,self.tokenizer,self.length)
        with self.lock:probs=np.asarray(self.model(array,training=False))
        if not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(axis=1),1,atol=1e-4):raise RuntimeError('invalid_model_probabilities')
        results=[]
        for prob,stat in zip(probs,stats):
            i=int(np.argmax(prob));score=float(prob[i]);warnings=[]
            if score<self.threshold:warnings.append('low_model_score')
            if stat['truncated']:warnings.append('input_truncated')
            if stat['oov_ratio']>.5:warnings.append('high_oov_ratio')
            results.append({'label':self.labels[i],'model_score':score,'scores':dict(zip(self.labels,map(float,prob))),'needs_review':bool(warnings),'warnings':warnings,**stat})
        return results

import sqlite3,json
from pathlib import Path
class Audit:
    def __init__(self,path):
        self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS predictions(request_id TEXT PRIMARY KEY,created_at TEXT DEFAULT CURRENT_TIMESTAMP,model_version TEXT,batch_size INTEGER,duration_s REAL,labels_json TEXT,review_count INTEGER)")
            c.execute("CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY,request_id TEXT REFERENCES predictions(request_id),item_index INTEGER,label TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP,UNIQUE(request_id,item_index))")
    def connect(self):
        c=sqlite3.connect(self.path,timeout=10);c.execute('PRAGMA foreign_keys=ON');return c
    def save(self,request_id,version,results,duration):
        with self.connect() as c:c.execute('INSERT INTO predictions(request_id,model_version,batch_size,duration_s,labels_json,review_count) VALUES(?,?,?,?,?,?)',(request_id,version,len(results),duration,json.dumps([r['label'] for r in results]),sum(r['needs_review'] for r in results)))
    def feedback(self,request_id,index,label):
        with self.connect() as c:
            row=c.execute('SELECT batch_size FROM predictions WHERE request_id=?',(request_id,)).fetchone()
            if not row or index>=row[0]:raise ValueError('unknown_prediction_item')
            c.execute('INSERT INTO feedback(request_id,item_index,label) VALUES(?,?,?) ON CONFLICT(request_id,item_index) DO UPDATE SET label=excluded.label',(request_id,index,label))
    def stats(self):
        with self.connect() as c:
            row=c.execute('SELECT COUNT(*),COALESCE(SUM(batch_size),0),COALESCE(SUM(review_count),0),COALESCE(AVG(duration_s),0) FROM predictions').fetchone()
            labels={}
            for (text,) in c.execute('SELECT labels_json FROM predictions ORDER BY created_at DESC LIMIT 1000'):
                for label in json.loads(text):labels[label]=labels.get(label,0)+1
            return dict(zip(['requests','predictions','review_flags','mean_request_duration_s'],row))|{'recent_label_counts':labels,'feedback_items':c.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]}

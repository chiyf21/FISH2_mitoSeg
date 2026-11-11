"""Shared lease queue, spatial order, finite retries, atomic SQLite claims."""
import argparse,json,sqlite3,time,secrets,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--port',type=int,default=18084);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);db=out/'queue.sqlite';rows=json.loads(Path(a.manifest).read_text())
    with sqlite3.connect(db) as c:
        c.execute('PRAGMA journal_mode=WAL');c.execute('CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY,payload TEXT,state TEXT,worker TEXT,attempt INTEGER DEFAULT0,updated REAL,result TEXT,priority INTEGER)'.replace('DEFAULT0','DEFAULT 0'))
        for priority,r in enumerate(rows):c.execute('INSERT OR IGNORE INTO jobs(id,payload,state,updated,priority) VALUES(?,?,?,?,?)',(r['index'],json.dumps(r),'pending',time.time(),priority))
    token=secrets.token_hex(24);access=out/'queue_access.json';access.write_text(json.dumps(dict(port=a.port,token=token)));access.chmod(0o600)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if data.get('token')!=token:self.send_error(403);return
            with sqlite3.connect(db,timeout=30) as c:
                if self.path=='/claim':
                    c.execute('BEGIN IMMEDIATE')
                    # Never infer worker death from time alone. Recovery must
                    # first verify the owner process exited before resetting it.
                    found=c.execute("SELECT id,payload FROM jobs WHERE state='pending' AND attempt<3 ORDER BY priority LIMIT1".replace('LIMIT1','LIMIT 1')).fetchone()
                    if found:c.execute("UPDATE jobs SET state='running',worker=?,attempt=attempt+1,updated=? WHERE id=?",(data['worker'],time.time(),found[0]))
                    result=dict(job=json.loads(found[1]) if found else None)
                elif self.path in ('/complete','/fail'):
                    state='done' if self.path=='/complete' else 'failed';c.execute('UPDATE jobs SET state=?,result=?,updated=? WHERE id=? AND worker=? AND state=?',(state,json.dumps(data.get('result')),time.time(),data['index'],data['worker'],'running'));result=dict(ok=True)
                else:self.send_error(404);return
                counts=dict(c.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall());result['counts']=counts
            (out/'progress.json').write_text(json.dumps(dict(counts=counts,time=time.time())))
            b=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    print(json.dumps(dict(port=a.port,jobs=len(rows))),flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()

"""Full-corpus dense and hybrid controls, sharing chunks and answer budgets."""
import json
from pathlib import Path

from finrate_poc.retrieval import CpuEmbedding
from finrate_poc.runtime import check_disk,file_digest
from .workspace import connect,keyword_query


class FullIndex:
    def __init__(self,folder):
        import numpy as np
        self.np=np;self.folder=Path(folder)
        manifest=json.loads((self.folder/'manifest.json').read_text())
        for name,sha in manifest['files'].items():
            if file_digest(self.folder/name)!=sha:raise ValueError('Index bytes changed')
        self.db=connect(self.folder/'chunks.sqlite',True)
        self.ids=json.loads((self.folder/'ids.json').read_text())
        self.matrix=np.load(self.folder/'vectors.npy',mmap_mode='r')
        self.embed=CpuEmbedding()
        if self.matrix.shape!=(len(self.ids),384):raise ValueError('Index shape mismatch')

    def search(self,question,*,method,k):
        check_disk();np=self.np
        scores=self.matrix@self.embed.encode([question])[0]
        order=np.argsort(-scores,kind='stable')[:max(200,k)]
        ranked=[(self.ids[int(i)],float(scores[i])) for i in order]
        if method=='hybrid':
            query=keyword_query(question)
            sparse=[] if query is None else self.db.execute('SELECT id,bm25(chunk_fts) AS score FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY score,id LIMIT 200',(query,)).fetchall()
            fused={}
            for results in [ranked,[(r[0],r[1]) for r in sparse]]:
                for rank,(ident,_) in enumerate(results,1):fused[ident]=fused.get(ident,0)+.5/(60+rank)
            ranked=sorted(fused.items(),key=lambda item:(-item[1],item[0]))
        elif method!='embedding':raise ValueError('Unknown retrieval baseline')
        return [{**dict(self.db.execute('SELECT * FROM chunks WHERE id=?',(ident,)).fetchone()),'score':score}
                for ident,score in ranked[:k]]

    def close(self):self.db.close()

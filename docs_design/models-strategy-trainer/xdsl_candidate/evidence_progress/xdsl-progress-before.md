Verbatim diagnostic driver; run with the recorded research environment from the repository root.

```text
import asyncio,hashlib,importlib,json,subprocess,sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
base='docs_design.models-strategy-trainer.xdsl_candidate.'
m=importlib.import_module(base+'mixed_adapter')
c=importlib.import_module(base+'mixed_contract')
p=importlib.import_module(base+'prepare')
e=importlib.import_module(base+'pair_extension')
f=importlib.import_module(base+'fixture')
class ProbeFinished(RuntimeError): pass
async def progress():
    calls=0; heartbeats=[]
    def reject(key,revision):
        nonlocal calls
        calls+=1
        if calls==1: asyncio.get_running_loop().call_soon(lambda:heartbeats.append(calls))
        if calls==20000: raise ProbeFinished('finite all-excluded progress probe')
        return False
    registry=m.mixed_registry(); bindings=dict(registry.bindings)
    bindings['dynamic_filter_v1']=replace(bindings['dynamic_filter_v1'],invoke=reject)
    report=await m.prepare_mixed(c.accept_mixed(c.build_mixed()),replace(registry,bindings=bindings)).publish().run()
    return {'filter_calls':calls,'first_heartbeat_at':heartbeats[0] if heartbeats else None,'status':report['status'],'released':report['released']}
async def outcomes():
    original=e.PackContract.describe
    def describe(self,op): return replace(original(self,op),outcomes=('negative_input',))
    e.PackContract.describe=describe
    try:
        registry=e.extension_registry(); bindings=dict(registry.bindings)
        bindings['pack_pair_v1']=replace(bindings['pack_pair_v1'],outcomes=('negative_input',))
        ready=p.prepare(p.accept(e.build_extension()),replace(registry,bindings=bindings))
        report=(await f.execute(ready)).report
        pack=next(call for call in report.tickets[0].calls if call.origin=='pair.pack')
        return {'declared':['negative_input'],'observed':pack.outcome,'status':report.status}
    finally: e.PackContract.describe=original
retirement='''import importlib,json
b="docs_design.models-strategy-trainer.xdsl_candidate."
r=importlib.import_module(b+"runtime"); p=importlib.import_module(b+"prepare"); f=importlib.import_module(b+"fixture")
s=r.ScalarSession(p.prepare(p.accept(f.build()),f.registry()))
t=r._Ticket(0,0,0,0,1); s._active[0]=t; s._clock=1
s._retire(t,"abandoned")
print(json.dumps({"outstanding":s.outstanding,"retired":len(s._completed)}))
'''
async def main():
    result={'reviewed_commit':'504dcf88dcdb2b76218e19af89117f7f7de22c59','versions':{'python':sys.version,'xdsl':version('xdsl'),'torch':version('torch'),'gil_enabled':sys._is_gil_enabled()},'exclusion':await progress(),'outcomes':await outcomes()}
    child=subprocess.run([sys.executable,'-O','-c',retirement],capture_output=True,text=True,check=True,timeout=15)
    result['optimized_retirement']=json.loads(child.stdout)
    root=Path('docs_design/models-strategy-trainer/xdsl_candidate')
    result['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('mixed_adapter.py','payloads.py','runtime.py','prepare.py')}
    Path('/tmp/xdsl-progress-before.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
asyncio.run(main())
```

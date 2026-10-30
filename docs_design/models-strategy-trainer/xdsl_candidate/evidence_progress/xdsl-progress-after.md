Verbatim diagnostic driver; run with the recorded research environment from the repository root.

```text
import asyncio,importlib,json,subprocess,sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
b='docs_design.models-strategy-trainer.xdsl_candidate.'
a=importlib.import_module(b+'mixed_adapter'); c=importlib.import_module(b+'mixed_contract'); p=importlib.import_module(b+'prepare'); e=importlib.import_module(b+'pair_extension')
async def progress(admit):
    calls=0; pulse=[]; feedback_at=[]
    def predicate(key,revision):
        nonlocal calls
        calls+=1
        if calls==1: asyncio.get_running_loop().call_soon(lambda:pulse.append(calls))
        if revision and not feedback_at: feedback_at.append(calls)
        if calls==20_000: raise RuntimeError('finite excluded-stretch probe complete')
        return admit and key==0
    registry=a.mixed_registry(); bindings=dict(registry.bindings); bindings['dynamic_filter_v1']=replace(bindings['dynamic_filter_v1'],invoke=predicate)
    run=a.prepare_mixed(c.accept_mixed(c.build_mixed()),replace(registry,bindings=bindings)).publish(); report=await run.run()
    if not pulse or pulse[0]>=20_000: raise RuntimeError('heartbeat starved')
    if admit and (not feedback_at or report['source']['revision']!=1): raise RuntimeError('real feedback starved')
    return {'filter_calls':calls,'first_heartbeat_at':pulse[0],'first_feedback_observed_at':feedback_at[0] if feedback_at else None,'steps':report['steps'],'feedback_revision':report['source']['revision'],'status':report['status'],'released':report['released'],'source_failures':{key:[str(error) for error in a.causes(feed.failure)] for key,feed in run.ready.state.feeds.items()}}
async def cancellation():
    calls=0; cancel_at=[]
    def cancel(): cancel_at.append(calls); task.cancel()
    def predicate(key,revision):
        nonlocal calls
        calls+=1
        if calls==1: asyncio.get_running_loop().call_soon(cancel)
        if calls==20_000: raise RuntimeError('finite cancellation fallback reached')
        return False
    registry=a.mixed_registry(); bindings=dict(registry.bindings); bindings['dynamic_filter_v1']=replace(bindings['dynamic_filter_v1'],invoke=predicate)
    run=a.prepare_mixed(c.accept_mixed(c.build_mixed()),replace(registry,bindings=bindings)).publish(); task=asyncio.create_task(run.run())
    try: await task
    except asyncio.CancelledError: pass
    else: raise RuntimeError('cancellation did not propagate')
    if calls>=20_000 or not run.last_report['released']: raise RuntimeError('cancellation did not settle promptly')
    return {'filter_calls':calls,'cancellation_callback_at':cancel_at[0],'status':run.last_report['status'],'released':run.last_report['released']}
def missing_outcomes():
    original=e.PackContract.describe
    def describe(self,op): return replace(original(self,op),outcomes=('negative_input',))
    e.PackContract.describe=describe
    try:
        try: p.accept(e.build_extension())
        except p.Rejected as error: return {'status':'rejected_at_acceptance','message':str(error)}
        raise RuntimeError('missing baseline outcomes accepted')
    finally: e.PackContract.describe=original
optimized='''import asyncio,importlib,json,sys
from dataclasses import asdict
b="docs_design.models-strategy-trainer.xdsl_candidate."
p=importlib.import_module(b+"prepare"); f=importlib.import_module(b+"fixture"); r=importlib.import_module(b+"runtime")
if sys.argv[1]=="extension":
 e=importlib.import_module(b+"pair_extension"); build,registry=e.build_extension,e.extension_registry
else: build,registry=f.build,f.registry
async def main():
 s=r.ScalarSession(p.prepare(p.accept(build()),registry(),fuse=True)); task=asyncio.create_task(s.run())
 await s.wait_for(lambda:s.feedback_count>=4); s.stop(); report=await task
 print(json.dumps({"optimized":not __debug__,"outstanding":s.outstanding,"active_calls":s.active_calls,"report":asdict(report)}))
asyncio.run(main())
'''
async def main():
    result={'versions':{'python':sys.version,'xdsl':version('xdsl'),'torch':version('torch'),'gil_enabled':sys._is_gil_enabled()},'quantum':a.SELECTION_QUANTUM,'all_excluded':await progress(False),'one_admitted_then_excluded':await progress(True),'cancellation':await cancellation(),'outcomes':missing_outcomes(),'optimized_sessions':{}}
    for mode in ('base','extension'):
        value=subprocess.run([sys.executable,'-O','-c',optimized,mode],text=True,capture_output=True,check=True,timeout=15)
        report=json.loads(value.stdout)
        if not report['optimized'] or report['outstanding'] or report['active_calls']: raise RuntimeError('optimized ownership mismatch')
        if mode=='base': importlib.import_module(b+'conformance').check_b(report['report'])
        else: importlib.import_module(b+'extension_conformance').check_extension(report['report'])
        result['optimized_sessions'][mode]=report
    Path('docs_design/models-strategy-trainer/xdsl_candidate/evidence_progress/after.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='optimized_sessions'},indent=2))
asyncio.run(main())
```

Verbatim diagnostic driver; run with the recorded research environment from the repository root.

```text
import asyncio,importlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path.cwd()))
b='docs_design.models-strategy-trainer.xdsl_candidate.'
a=importlib.import_module(b+'mixed_adapter'); tests=importlib.import_module(b+'test_mixed_runtime')
async def main():
    reports={}
    for lowering in ('calls','inline'):
        for fuse in (False,True):
            for fail in (False,True):
                key=f'{lowering}_{"fused" if fuse else "unfused"}_{"cancel_physical_failure" if fail else "normal_stop"}'
                original=a.MixedRun.report
                def observe(self,failure=None):
                    report=original(self,failure)
                    unit=next(iter(self.ready.state.image.units.values()))
                    reports[key]={'report':report,'actual_weight':float(unit.parameters[0].detach()),'owner_source_epochs':[next(item for item in outcome.input_handoffs if item.channel=='mixed.right').provenance[0].numerical_epoch for outcome in sorted(unit.outcomes,key=lambda value:value.invocation)]}
                    return report
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(a.MixedRun,'report',observe)
                    await tests.test_live_feedback_and_model_ownership_do_not_depend_on_observation_archives(patch,lowering,fuse,fail)
    root=Path('docs_design/models-strategy-trainer/xdsl_candidate/evidence_progress')
    (root/'archive_matrix.json').write_text(json.dumps({'gil_enabled':sys._is_gil_enabled(),'cases':reports},indent=2)+'\n')
    print(json.dumps({'cases':len(reports),'all_test_assertions':'passed','gil_enabled':sys._is_gil_enabled()}))
asyncio.run(main())
```

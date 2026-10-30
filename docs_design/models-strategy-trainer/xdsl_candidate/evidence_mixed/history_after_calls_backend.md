```python
async def body(region, selected, input_scope):
    choice = selected.payload if selected is not None else None
    frame = await enter('contribution', region, selected, input_scope)
    # accepted source 'admit'
    v_0_0, v_0_1 = await work_0(frame)
    # accepted source 'retain'
    v_1_0 = await work_1(frame)
    # accepted source 'objective'
    v_2_0 = await work_2(frame, v_0_0, v_0_1, v_1_0)
    # accepted source 'differentiate'
    v_3_0 = await work_3(frame, v_2_0, v_1_0)
    # accepted source 'release'
    await work_4(frame, v_1_0)
    completed('contribution')
    frame = await enter('finish', region, selected, input_scope)
    # accepted source 'advance'
    await work_5(frame, v_3_0)
    # accepted source 'learned'
    await work_6(frame, v_0_0)
    completed('finish')
    return frame.invocation
```

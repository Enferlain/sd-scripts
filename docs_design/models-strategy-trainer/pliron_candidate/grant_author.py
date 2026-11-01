"""Author the bounded C fixture in Pliron; no implementations are installed."""

from __future__ import annotations

from .author import attributes

PHASES = ("backward", "temporary_edit", "intermediate_reset", "backward", "restore")


def build(
    *,
    initial: float = 0.8,
    learning_rate: float = 0.05,
    clip_norm: float | None = None,
    prefix: str = "",
    participant: str = "model",
    unit: str = "weights",
) -> str:
    def a(source: str, **values: str) -> str:
        return attributes(source_id=prefix + source, **values)

    phases = []
    previous = "access"
    for index, phase in enumerate(PHASES):
        name = f"p{index}"
        phases.append(f"{name} = ga.phase ({previous}) [] {a(f'phase.{index}.{phase}', phase=phase)} : <(ga.access) -> (ga.access)>")
        previous = name
    phases.append(f"ga.yield ({previous}) [] {a('selected.handback')} : <(ga.access) -> ()>")
    return f"""ga.session () [] {a("grant-session", version="1", failure="drain_entered_preserve_uncertain")} : <() -> ()> {{
  ^scope():
    model = ga.model () [] {a("model.binding", participant=participant, initial=repr(initial), treatment="trainable", implementation="scale_v1")} : <() -> (ga.model)>;
    unit = ga.unit (model) [] {a("owner.weights", unit=unit, parameter="weight")} : <(ga.model) -> (ga.unit)>;
    profile = ga.offer () [] {a("owner.profile", profile="restored-two-pass-v1", delegated="backward,temporary_edit,intermediate_reset,restore", retained="clipping,final_advancement,final_reset")} : <() -> (ga.profile)>;
    handback = ga.execute (model, unit, profile) [] {a("selected-two-pass", implementation="two_pass_v1", completion="physical_return_then_owner_handback_verification", handback="restored_current_final_gradient_then_owner_verifies")} : <(ga.model, ga.unit, ga.profile) -> (ga.handback)> {{
      ^selected(access: ga.access):
        {"; ".join(phases)}
    }};
    advanced = ga.advance (handback, unit) [] {a("retained-final-step", learning_rate=repr(learning_rate), clip_norm="none" if clip_norm is None else repr(clip_norm), authority="retained_owner_only")} : <(ga.handback, ga.unit) -> (ga.done)>;
    ga.finish (advanced) [] {a("grant.finish")} : <(ga.done) -> ()>
}}
"""

# reasoning (Project 2)

How to complement them.

**build**  interaction evidence -> an adapter on the LLM
**serve**  applied during generation, alongside knowledge

## What makes this different from knowledge

Knowledge is retrieved state. This is parameter adaptation. Your thesis calls them Level A
and Level B, and the point of keeping them separate slots is that the entity model stays
portable while the adapter is disposable and model-specific.

## The open problem

**Complementation has no metric yet.** Mimicry does: does it sound like them, and `ims`
already scores it. "Offers thoughts they would miss" is P2's central claim and nothing
currently would count as evidence it worked. Without a score there is no steering, and no
auto-promotion.

## Status

Not built. Deliberately deferred. Do not build P2's personalization machinery during P1.

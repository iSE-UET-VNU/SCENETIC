RUBRICS = """
Evaluate the scenario on the following six criteria, using a score from 0 to 5 for each one.

1. Kinematic Plausibility
Assess whether agent motion is physically plausible across frames.
Check whether:
- position changes are consistent with velocity
- rotation is consistent with direction of travel
- acceleration or deceleration changes are not unnaturally abrupt
- angular velocity is consistent with turning behavior
- pedestrians move like pedestrians and vehicles move like vehicles

Scoring guide:
- 5: Motion is smooth, continuous, and physically consistent throughout the sequence.
- 4: Mostly physically plausible, with only minor imperfections.
- 3: Noticeable issues exist, but the sequence remains somewhat plausible overall.
- 2: Several motion patterns are unrealistic or poorly justified.
- 1: Motion is highly unrealistic, with major discontinuities or implausible changes.
- 0: Motion is physically impossible or severely inconsistent across frames.

2. Map and Junction Consistency
Assess whether actor trajectories and events are consistent with the available road structure and junction configuration.
Check whether:
- agent behavior matches the available junction entries
- no actor behaves as if a nonexistent road branch exists
- trajectories and headings fit the junction topology
- actor presence near the junction is consistent with the stated junction context

Scoring guide:
- 5: All relevant actions and trajectories align clearly with the road environment and junction context.
- 4: The scenario is largely consistent with the junction and road structure.
- 3: Some behaviors are weakly supported by the junction context, but the scenario remains partially interpretable.
- 2: Multiple behaviors appear mismatched with the available road layout.
- 1: Major inconsistencies exist between actor behavior and junction topology.
- 0: The scenario strongly contradicts the road or junction structure.

3. Agent Behavioral Realism
Assess whether vehicles and pedestrians behave in a believable and context-appropriate manner.
Check whether:
- vehicles follow plausible paths and react sensibly to nearby agents
- vehicles approach, stop, or proceed in a coherent and goal-directed way
- pedestrians stand, walk, cross, or wait in believable ways
- agents do not look artificially inserted just to create difficulty

Scoring guide:
- 5: Agent behavior is consistently believable, goal-directed, and appropriate to the traffic context.
- 4: Most agent behavior is realistic, with only minor stiffness or inefficiency.
- 3: Behavior is mixed: some actions are believable, while others feel awkward or weakly justified.
- 2: Several actors behave unnaturally or appear inserted only to create difficulty.
- 1: Behavior is highly unrealistic and difficult to justify in context.
- 0: Agent behavior is entirely implausible or contradictory.

4. Interaction Realism
Assess whether interactions among ego, vehicles, and pedestrians develop naturally and coherently over time.
Check whether:
- approach and separation patterns are believable
- crossing, yielding, blocking, conflict, and merging behavior are realistic
- reactions occur at plausible times
- near-collisions or collisions emerge from believable interaction dynamics

Scoring guide:
- 5: Interactions are natural, well-timed, and strongly supported by the preceding context.
- 4: Interactions are mostly realistic and contextually appropriate.
- 3: Some interactions are realistic, but others are awkward or insufficiently supported.
- 2: Several interactions feel forced, poorly timed, or weakly motivated.
- 1: Interactions are largely implausible or disconnected from the surrounding context.
- 0: Interactions are absent, contradictory, or entirely unrealistic.

5. Temporal Consistency
Assess whether the full sequence remains coherent across frames.
Check whether:
- actor identity remains stable over time
- actors do not appear or disappear without explanation
- trajectories remain continuous
- later frames follow naturally from earlier frames
- collision states, if present, are followed by plausible consequences

Scoring guide:
- 5: The full sequence is temporally coherent, continuous, and causally well-connected.
- 4: The sequence is mostly continuous and temporally consistent.
- 3: The sequence is partly coherent, but notable continuity issues remain.
- 2: Several discontinuities or poorly connected transitions reduce realism.
- 1: Major continuity problems frequently break the scenario.
- 0: The sequence is temporally incoherent and cannot be interpreted as a consistent event.

6. Realistic Edge-case Formation
Assess whether rare or safety-critical events emerge plausibly from prior context, rather than appearing artificially introduced.
Check whether:
- the edge case is properly built up by preceding frames
- the hazardous event emerges from realistic traffic interaction
- the sequence does not feel staged or adversarial in an artificial way
- the event is rare but still believable

Scoring guide:
- 5: The event is rare or difficult, yet emerges naturally and convincingly from the preceding context.
- 4: The event is largely plausible and reasonably supported by the scenario buildup.
- 3: The event is somewhat plausible, but still shows signs of artificial construction.
- 2: The event has limited realism and feels noticeably forced.
- 1: The event is highly artificial and poorly supported by earlier context.
- 0: The safety-critical event is entirely contrived or implausible.

Evaluation instructions:
- Judge the full sequence, not isolated frames only.
- Base your judgment on continuity across time.
- Be strict about physical inconsistency, abrupt unexplained changes, and implausible interaction buildup.
- Do not assume missing facts in favor of realism.
- If the available evidence is insufficient, reflect that uncertainty in the explanation and avoid giving overly high scores.
- Do not give high scores just because the scenario is complex, dangerous, or difficult.
- Focus on whether the sequence is believable as a real traffic situation from start to finish.
"""

OUTPUT_FORMAT = """
Output format:
Return your answer strictly in valid JSON with the following structure:

{
  "kinematic_plausibility": {
    "score": 0,
    "reasoning": ""
  },
  "map_and_junction_consistency": {
    "score": 0,
    "reasoning": ""
  },
  "agent_behavioral_realism": {
    "score": 0,
    "reasoning": ""
  },
  "interaction_realism": {
    "score": 0,
    "reasoning": ""
  },
  "temporal_consistency": {
    "score": 0,
    "reasoning": ""
  },
  "realistic_edge_case_formation": {
    "score": 0,
    "reasoning": ""
  },
  "overall_realism_score": 0,
  "probability": float,
  "confidence": float
}

Additional output requirements:
- Each score must be an integer from 0 to 5.
- Keep each reasoning field concise but specific.
- The overall_realism_score must also be an integer from 0 to 5.
- The overall score should reflect the full scenario, not the average mechanically.
- Do not include markdown.
- Do not include any text outside the JSON object.
"""
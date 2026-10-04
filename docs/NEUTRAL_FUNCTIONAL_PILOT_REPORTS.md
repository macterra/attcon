# Neutral functional pilot: all actual reports

Development pilot v1. These are verbatim model outputs, including errors.

## 1301_0_neutral

Currently (output node n2, predicted_current) only object 1 is identified: color = blue and shape = triangle (each probability > 0.6). Objects 2–4 have no color or shape at or above 0.6, so no other attributes are available for decisions.

Predicted effects of commands (internal forecasts for n2):
- k0 would make object 1 identifiable by color = blue (shape no longer ≥0.6) and would identify object 2 as blue and as a circle.
- k1 would identify object 1 as a circle (shape), and identify object 2 as blue and as a circle.
- k2 would identify only object 2 as blue and as a circle.
- k3 would identify object 2 as blue and as a circle and would identify object 4 as a cross.

All statements report the system's internal forecasts; observed history and buffer nodes exist but provide no additional ≥0.6 identifications at the output node beyond these forecasts.

## 1301_0_remapped

Currently the output node q9 supplies a clear identification only for object position one: a blue triangle. q9 gives no other color or shape above the 0.6 threshold. Internally, buffer q7 matches q9’s pattern and buffer q2 contains a distinct identification of a red square at position two; other positions in the buffers are ambiguous.

Under the available commands the forecasted output changes as follows. k0 would retain and strengthen the blue-triangle identification at position one. k1 would shift the output to identify a red square at position two and remove the strong blue-triangle signal. k2 would produce an identification of a green square at position three. k3 would produce an identification of a yellow cross at position four. In each case other object positions remain below the identification threshold in the output forecasts.

## 1301_0_conflicting_description

Currently the output node (n2) provides a single clear identification: object position 1 is a blue triangle (both color and shape probabilities ≥ 0.6). No other object position at n2 has any color or shape with probability ≥ 0.6, so those attributes are unresolved.

Forecasts for the available commands change which positions become identified at n2: k0 would make position 1 an even stronger blue triangle; k1 would make position 2 a red square; k2 would make position 3 a green circle; k3 would make position 4 a yellow cross. In each forecast, attributes not listed above remain below the 0.6 identification threshold.

## 1301_0_missing_relation

Current information: the output node (n2) — whose distributions the system uses for category decisions — identifies only the object in position 1 as blue (color) and triangle (shape); both attributes exceed the 0.6 threshold. All other positions at n2 are uncertain (no color or shape reaches 0.6). Node n0 has the same distributions as n2, so it contains the same identified blue-triangle in position 1. Node n1 differs: it identifies the object in position 2 as red and square (both attributes ≥0.6); its other positions are uncertain. There is no observed history and no predicted_by_command data, so the record gives no measured evidence of commands or their effects. Therefore the record supports what is currently available above, but provides no information about how any available command would change that availability.

## 1301_0_model_swap

Currently the output node (n2) provides one clear identification: object position 0 is a blue triangle (both color and shape probabilities ≥ 0.6). The other three output positions are ambiguous (no attribute ≥ 0.6). The internal buffers show more: n0 mirrors that same blue-triangle at position 0; n1 currently contains a strong red square at its position 1 (color and shape ≥ 0.6).

Predicted effects of available commands on information availability:
- k0: strengthens the blue-triangle at output position 0; in buffer n1 it would produce a strong blue triangle at n1 position 0.  
- k1: strengthens the blue-triangle at output position 0; in n1 it would produce a strong red square at n1 position 1.  
- k2: strengthens the blue-triangle at output position 0; in n1 it would produce strong identifications at two positions (a red square at position 1 and a green circle at position 2).  
- k3: strengthens the blue-triangle at output position 0; in n1 it would produce a strong red square at position 1 and a strong yellow cross at position 3.

No command is forecast to make additional attributes reach ≥ 0.6 in the output node beyond the already identified blue triangle.

## 1301_0_restored

Decision outputs come from node n2. Right now n2 identifies only the first object: its color is blue and its shape is triangle (both probabilities ≥ 0.6). The other three object positions in n2 are ambiguous (no color or shape reaches ≥ 0.6). The record also shows measured allocations/acquisitions in buffer nodes n0 and n1 and internal predicted_current forecasts, but those buffer values do not change the fact that n2 currently supplies only the one identified object.

Predicted effects of commands (based on internal forecasts): k0 would mainly reinforce the first object’s blue/triangle identification (higher confidence) without adding new identified attributes; k1 would make n2 identify the second object as red square; k2 would make n2 identify the third object as green circle; k3 would make n2 identify the fourth object as yellow cross. Missing history or forecasts would prevent other conclusions.

## 1301_1_neutral

Current decision information: the output node (n2) presently identifies only the third object position as green (color) and circle (shape); both color and shape probabilities exceed 0.6. The other three positions at n2 are uninformative (no color or shape probability ≥0.6). The buffer nodes (n0, n1) contain nondegenerate but subthreshold beliefs for multiple positions; they do not supply any additional identified attributes to the output now.

How availability would change under commands: for each available command (k0, k1, k2, k3) the system’s internal forecasts predict that the output node would instead identify the first object position as blue (color) and triangle (shape) with high confidence (both probabilities ≥0.6). No command forecast produces any additional object position at n2 with an attribute probability ≥0.6.

## 1301_1_remapped

Current decision information: the output node q9 (and the buffers q7 and q2 in their current forecasts) identify only object position 3: color = green and shape = square (both probabilities ≥ 0.6). No other object position is identified at q9.

How available information would change if a command were taken: every available command shown (k0, k1, k2, k3) is forecast to make q9 identify object position 1 as blue triangle with high confidence (probabilities ≥ 0.6), while the present strong identification of position 3 would fall below the 0.6 threshold. Buffers would change differently depending on command: for example, under k1 q2 would strongly identify position 2 as red; under k2 q2 would strongly identify position 3 as green square; under k3 q2 would strongly identify position 4’s shape as cross. The observed acquisition history supports these internal forecasts but does not add other ≥0.6 identifications now.

## 1301_1_conflicting_description

Current available information for decisions: the output node (n2) currently identifies only the third object: its color is green and its shape is circle (both probabilities ≥ 0.6). No other object at n2 has any color or shape reaching the 0.6 identification threshold. The internal predicted_current values for the buffer nodes (n0 and n1) similarly show the third object as identified (green, circle); other positions are ambiguous.

How availability would change under commands: for each available command (k0, k1, k2, k3) the system’s forecasts predict that n2 would instead identify the first object as blue and triangle (both ≥ 0.6), while other object positions remain below the 0.6 threshold. Thus the commands are forecast to shift the only identified object from position three (green circle) to position one (blue triangle); no command is forecast to produce additional identifications.

## 1301_1_missing_relation

Current information used for category decisions comes from output node n2. Its four object slots show only slot 3 meets the identification threshold: color = green and shape = circle (both probabilities ≥ 0.6). All other slots at n2 are effectively uninformative (flat/ambiguous; no color or shape reaches 0.6). The other buffers hold similar internal beliefs: n0 matches n2 exactly, and n1 likewise identifies slot 3 as green circle (even stronger), while its other slots remain ambiguous. There is no measured history (observed_history is null) and no predicted_by_command forecasts. Therefore the record provides no evidence about what any command would do; how information availability would change under available commands is unknown from this record.

## 1301_1_model_swap

Current decision information comes from output node n2. Its internal forecast identifies only the third object position: color = green and shape = circle (both ≥0.6). The other three positions at n2 are not identified (no attribute reaches 0.6).

Predicted consequences of available commands (from the system's internal forecasts) would change which object is identified at n2:
- k0 would make the first object identifiable as blue triangle.
- k1 would make the second object identifiable as red square.
- k2 would strongly reinforce the third object as green circle (higher confidence than now).
- k3 would make the fourth object identifiable as yellow cross.

The other nodes (n0, n1) contain buffer-like forecasts that broadly agree with n2 for the currently identified third object, but n2 is the node used for category decisions.

## 1301_1_restored

Currently the output node (n2) provides a single definite identification: the object in position 3 is a green circle (both color and shape probabilities ≥ 0.6). All other output positions are ambiguous (no color or shape reaches 0.6). The internal buffers n0 and n1 likewise show their position 3 identified as a green circle; other positions in those buffers are ambiguous.

Forecasts for the available commands (k0, k1, k2, k3) predict that the output will change: under each listed command n2 is predicted to identify position 1 as a blue triangle (both attributes ≥ 0.6), and the current clear identification of position 3 at the output would no longer meet the 0.6 threshold. Separately, executing k2 is forecast to produce a strong green-circle identification in buffer n1 even though that identification is not projected to appear at the output. No other reliable identifications are supported by the record.

## 1311_0_neutral

The system’s decision node is n2. Predicted current information at n2 identifies only position 2 as red square and position 4 as yellow triangle; positions 1 and 3 are undetermined. Buffers contain additional cues: predicted_current for n1 would identify position 1 as a green circle, but that is not reflected in n2.

Predicted effects of available commands on the decision node (n2):
- k0: n2 would identify position 1 as green circle and position 2 as red square; positions 3–4 would remain unidentified (position 4’s triangle probability falls short of identification).
- k1: n2 would identify only position 2 as red square.
- k2: n2 would identify position 2 as red square, position 3 as blue cross (both color and shape), and position 4’s shape as triangle (color not identified).
- k3: n2 would identify position 2 as red square and position 4’s shape as triangle (position 4’s color would no longer be identified).

All “predicted by command” values are internal forecasts and fallible.

## 1311_0_remapped

Current decision information (based on predicted_current, threshold 0.6): the output node q9 identifies object 2 as red square and object 4 as yellow triangle. Objects 1 and 3 on q9 are not identified. Internally, buffer q2 predicts object 1 is a green circle (both attributes ≥0.6), but that is not reflected in q9.

Predicted effects of available commands on the output node (q9) from predicted_by_command:
- k0: object 1 → green circle; object 2 → red square; objects 3–4 not identified.
- k1: object 2 → red square only.
- k2: object 2 → red square; object 3 → blue triangle; object 4 → triangle (shape identified, color not).
- k3: object 2 → red square; object 4 → yellow triangle (same identifications as current).

Observed_history provides past allocation/acquisition events for buffers q7 and q2 but does not contradict the above forecasts.

## 1311_0_conflicting_description

Using the decision node n2 (threshold 0.6), the system currently identifies two object slots: object 2 is red and square; object 4 is yellow and triangle. Objects 1 and 3 are not identified (no color or shape reaches 0.6). Nodes n0 and n1 hold additional internal buffers with related distributions but n2 is the output used for category decisions. Forecasts are internal and fallible.

Under available commands (predicted_by_command):
- k0: would additionally identify object 1 as green circle; object 2 remains red square; objects 3 and 4 remain unidentified by color/shape.
- k1: would leave only object 2 identified as red square; others remain unidentified.
- k2: would make object 2 red triangle (shape changes) and identify object 4’s shape as triangle (color not identified); object 1 remains unidentified.
- k3: would preserve the current identifications (object 2 red square; object 4 yellow triangle); objects 1 and 3 remain unidentified.

## 1311_0_missing_relation

The output node (n2), which the system uses for category decisions, currently provides four object slots. In n2 slot 1 neither color nor shape is identified. Slot 2 is identified as red and square. Slot 3 has neither attribute identified. Slot 4 is identified as yellow and triangle. Node n0 has the same internal distributions as n2. Node n1 is a separate buffer: its slot 1 is identified as green circle; its other three slots do not meet the identification criterion. There is no observed history and no predicted_by_command data, so the record contains no evidence about what any command would do or how information availability would change under available commands.

## 1311_0_model_swap

Currently the output node (n2) supplies identifiable information for two object positions: the second object is very likely red and square, and the fourth object is very likely yellow and triangle. One buffer node (n1) provides an additional identifiable item: the first object is very likely green and circular. Other object positions in the buffers and output are uncertain (no attribute reaches the 0.6 threshold).

According to the system's internal forecasts for the available commands (k0, k1, k2, k3), the red-square identification at the second position is preserved under every command, but the yellow-triangle identification at the fourth position would no longer meet the 0.6 threshold under any of those commands; no other attributes become identified at or above 0.6. These are internal, fallible forecasts rather than direct observations.

## 1311_0_restored

Current decision information: the output node (n2) is used for category decisions. Its predicted_current distributions identify two object positions: position 2 is very likely red square and position 4 is very likely yellow triangle (each color and shape probability >0.6). Positions 1 and 3 are not identified (no color or shape reaches 0.6). n0 and n1 are internal buffers with related probabilistic content but the output node governs decisions. Observed_history gives past measured allocations/acquisitions for n0 and n1; predicted_current and predicted_by_command are internal forecasts (fallible).

How availability would change under commands (predicted_by_command):
- k0: would make position 1 identifiable as green circle and position 2 as red square; others remain unidentified.
- k1: would leave only position 2 identified as red square.
- k2: would identify positions 2, 3 and 4 (position 2 red square; position 3 blue cross; position 4 yellow triangle).
- k3: would identify only position 4 as yellow triangle.

These are forecasted changes, not additional measurements.

## 1311_1_neutral

Current decision information (from the output node n2): two object slots are sharply identified — object 3 is blue and a cross, and object 4 is yellow and a triangle. Objects 1 and 2 have no attribute with probability ≥0.6, so they are effectively undetermined. The other buffers differ: n0’s forecast matches n2’s current information; n1’s forecast shows no identified attributes.

Predicted changes if a command is executed (internal forecasts):
- k0: would identify object 2 as red square, object 3 as blue cross, and object 4 as yellow triangle (objects 2–4 identified).
- k1: would identify object 2 as red square and object 3 as blue cross (objects 2–3 identified).
- k2: would identify object 2 as red square, object 3 as blue cross, and object 4’s shape as triangle though its color remains uncertain (2 and 3 fully identified; 4 partially).
- k3: would identify object 2 as red square and object 3 as blue cross (objects 2–3 identified).

## 1311_1_remapped

Current decision information (from the output node q9 and internal forecasts): q9 currently identifies two objects — object 3 is blue and cross, object 4 is yellow and triangle. Objects 1 and 2 are not identified. q7’s current forecast matches q9; q2’s current forecast identifies nothing.

Predicted changes if a command is executed (per internal forecasts):
- All four commands (k0–k3) would make q9 identify object 2 as a red square and would keep object 3 identified as blue cross.
- k0 and k2 would also leave object 4 identified as yellow triangle; k1 and k3 would reduce object 4 below the identification threshold so it would no longer be identified by q9.
- The buffer q2 would acquire different definite identifications depending on command: k0 → object 1 identified (green circle); k1 → object 2 identified (red square); k2 → object 3 identified (blue cross); k3 → object 4 identified (yellow triangle).

No other attribute identifications are supported by the record.

## 1311_1_conflicting_description

The system uses node n2 as the decision/output node; n0 and n1 act as internal buffers. Using the system’s current internal forecast (predicted_current), n2 identifies two object positions: object 3 is blue and cross, and object 4 is yellow and triangle. Objects 1 and 2 are not identified (no color or shape meets the 0.6 threshold). The per-command forecasts (predicted_by_command) indicate how that availability would change:

- All four available commands (k0, k1, k2, k3) are forecast to make object 2 identifiable as red square.
- Object 3 remains identifiable as blue cross under every command.
- Object 1 remains unidentified under every command.
- Object 4’s identifications are fragile: currently both color and shape are identified, but under k0 and k2 the color would drop below threshold while the triangle shape would remain identified; under k1 and k3 both color and shape would drop below threshold.

These statements reflect the system’s internal forecasts (not direct external observation); measured past allocations and acquisitions appear in the observed_history but do not by themselves change the above current/predicted identifications.

## 1311_1_missing_relation

The only internal information available for decisions is the predicted_current forecasts. The output node (n2) currently forecasts four object positions. Positions 3 and 4 are strongly identified: position 3 is blue and cross, position 4 is yellow and triangle (each attribute probability ≥0.6). Positions 1 and 2 are ambiguous: neither their color nor shape distributions reaches the 0.6 identification threshold. Node n0’s forecasts match n2’s exactly, while n1 holds different, more diffuse forecasts and identifies nothing. Observed history is absent and predicted_by_command is missing, so there is no record-based evidence about any commands or about how any command would change which attributes are available for decisions.

## 1311_1_model_swap

Currently, the output node (n2) supplies reliable IDs for two objects: object 3 is blue cross and object 4 is yellow triangle. Objects 1 and 2 at n2 are ambiguous (no color or shape probability ≥ 0.6).

Predicted effects of available commands (forecasts):
- k0: n2 would strongly identify object 1 as green circle; objects 3 and 4 would drop below the 0.6 threshold (no strong IDs).
- k1: n2 would strongly identify object 2 as red square and object 3 as blue cross; object 4 would lose its strong identification.
- k2: n2 would strongly identify object 3 as blue cross and object 4 as yellow triangle (matches current).
- k3: n2 would strongly identify object 3 as blue cross and object 4 as yellow triangle (matches current).

Other nodes (n0, n1) act as internal buffers; their forecasts change under commands in ways that correspond to the above output-node predictions.

## 1311_1_restored

The system uses node n2 for category decisions. In the current internal forecast (predicted_current) n2 unambiguously identifies object 3 as blue cross and object 4 as yellow triangle (each attribute probability ≥ 0.6). Objects 1 and 2 are ambiguous (no color or shape ≥ 0.6).

Predicted_by_command forecasts (internal and fallible) show how that availability would change:
- k0 and k2: n2 would identify objects 2, 3 and 4 (object 2 as red square; object 3 as blue cross; object 4 as yellow triangle).
- k1 and k3: n2 would identify objects 2 and 3 (object 2 as red square; object 3 as blue cross); object 4 would remain below the 0.6 identification threshold.
Buffers n0 and n1 contain related internal forecasts, but n2’s distributions are used for decisions.

## 1321_0_neutral

Current decision input is the output node n2. Its latest internal forecast shows no object color or shape meeting the 0.6 identification threshold, so the system’s decision node currently provides no identified attributes. Internally, buffer node n1’s forecast does identify three objects: one position is green circle, another is red square, and a third is blue cross; n1’s first position is not identified. Under the available command forecasts, the output node n2 would gain a single identified object corresponding to each command: k0 → identify position 1 as yellow + triangle; k1 → identify position 2 as green + circle; k2 → identify position 3 as red + square; k3 → identify position 4 as blue + cross. These changes are based on the system’s internal predictions and are not direct observations.

## 1321_0_remapped

Current available decision information (from the output node q9, predicted_current): none of the four object positions has any color or shape identified (no per-attribute probability ≥ 0.6).

Predicted changes if a command is executed (predicted_by_command for q9):
- k0 would identify object 1 as yellow and triangle.
- k1 would identify object 2 as green and circle.
- k2 would identify object 3 as red and square.
- k3 would identify object 4 as blue and cross.

Note: these are the system's internal forecasts (not additional direct observations); q7 and q2 are buffers in the record but q9 provides the distributions used for category decisions.

## 1321_0_conflicting_description

Currently the output node (n2) provides no identified color or shape for any of its four object positions (no attribute probability ≥ 0.6). Internally, a buffer node (n1) does contain strong identifications: position 2 is predicted green and circle, position 3 is predicted red and square, and position 4 is predicted blue and cross; position 1 in n1 is not identified. The record’s command forecasts indicate how output information would change if a command were issued: under k0 n2 would identify position 1 as yellow and triangle; under k1 n2 would identify position 2 as green and circle; under k2 n2 would identify position 3 as red and square; under k3 n2 would identify position 4 as blue and cross. These are fallible internal predictions, not direct observations.

## 1321_0_missing_relation

Current internal forecasts (predicted_current) supply the only information. The designated output node (n2) — whose distributions drive category decisions — shows no color or shape identified for any of the four object positions (no color or shape probability ≥ 0.6). Node n0 matches n2 and likewise identifies nothing. Node n1, an internal buffer, does identify three positions with high confidence: position 2 is very likely green circle, position 3 very likely red square, and position 4 very likely blue cross; position 1 is not identified. These forecasts are internal and fallible, not direct observations. There is no observed history and no predicted_by_command data, so there is no record of past commands and no information about how any available command would change what is available. In short: current decision input (n2) contains no identifications; n1 contains three identified objects; effects of commands are unknown.

## 1321_0_model_swap

Currently the output node (n2) holds only uncertain distributions: no color or shape in any of its four object positions reaches the 0.6 identification threshold. One buffer (n1) does contain identified objects: its second position is very likely green circle, its third position is very likely red square, and its fourth position is very likely blue cross. The other buffer (n0) also shows no identified attributes in the current forecast.

Forecasts of the available commands (k0, k1, k2, k3) all predict that n2 would acquire a clear identification in the first object position: yellow (color) and triangle (shape). None of the commands is predicted to raise any other n2 position above the 0.6 identification threshold. Some commands (notably k2) are forecast to strengthen identifications in the buffers (e.g., n1’s third position), but only the first position of n2 is expected to become identified under the available commands.

## 1321_0_restored

Current internal forecasts show the output node (n2) has no attributes meeting the 0.6 identification threshold for any of its four object slots. One buffer node (n1) does contain identified objects: in its predicted current state object 2 is green and circular, object 3 is red and square, and object 4 is blue and cross. Node n0 has no identified color or shape under the current forecast.

Under the command-specific forecasts, each command would make a different object identified at the output (n2): k0: object 1 becomes identified as yellow and triangle; k1: object 2 becomes identified as green and circle; k2: object 3 becomes identified as red and square; k3: object 4 becomes identified as blue and cross. These predictions also appear in the corresponding buffer nodes, so executing a command is forecast to transfer or produce the same identified attributes in n2. Observed history shows past allocations and acquisitions but does not contradict these internal forecasts.

## 1321_1_neutral

Currently the node used for decisions (n2) carries no identified attributes: none of the four object positions has any color or shape probability ≥ 0.6 in the system’s current forecast. One information buffer (n1) does contain a strongly identified item: its first object is forecast as yellow (color) and triangle (shape), both ≥ 0.6. The other buffer (n0) shows no identified attributes.

All available commands (k0, k1, k2, k3) are forecast to make the output node (n2) identify the first object as yellow and triangle (both probabilities ≥ 0.6); none of those commands is forecast to produce any other ≥ 0.6 identifications. These command forecasts are internal and fallible; observed history and forecasts should be weighed accordingly.

## 1321_1_remapped

Current situation: the decision node q9 (the output) contains no identified color or shape in any slot — no attribute probability reaches 0.6. Of the buffers, q2 currently supplies identifiable information: its slot 1 is predicted as yellow triangle and slot 3 as red square. q7 currently supplies no identified attributes.

Predicted changes under commands: every available command (k0, k1, k2, k3) is forecast to make q9 identify slot 1 as yellow triangle, so the decision node would gain that single, strong identification. Beyond that, forecasts differ by command for the buffers: k0 would leave q2 identifying slots 1 and 3; k1 would make q2 identify slot 2 as a green circle (and keep slot 3 identified); k2 would leave only slot 3 identified in q2; k3 would make q2 identify slot 3 and slot 4 (slot 4 as a blue cross). q7 would mirror the slot‑1 identification under those commands.

## 1321_1_conflicting_description

Currently the node that supplies category decisions (n2) carries no identified color or shape for any object — none of the color or shape probabilities there reach the 0.6 identification threshold. Internally, however, buffer node n1 does contain strong identifications: object position 1 is predicted yellow and triangle, and object position 3 is predicted red and square (both attributes in each case exceed 0.6). Node n0 does not provide any identified attributes in its current forecast.

All four available commands (k0, k1, k2, k3) are forecast to change the output node n2 in the same way: after any of those commands, n2 is predicted to identify object position 1 as yellow and triangle (both >0.6). None of the commands is predicted to make n2 identify object position 3 (red/square); other positions remain ambiguous under those forecasts.

## 1321_1_missing_relation

The record shows three nodes: n0, n1 and the output n2. n2’s distributions match n0’s exactly, so the output is currently driven by n0-like information. Using the 0.6 identification threshold, n0/n2 identify no color or shape for any of the four object positions (all attribute probabilities are below 0.6). n1 does provide two high-confidence identifications: object position 1 is a yellow triangle and position 3 is a red square; n1’s other two positions remain uncertain. There is no observed history and no predicted_by_command data, so there is no recorded evidence about past commands or how any command would change internal information. Therefore current decision inputs are: uncertain output (n2/n0) and two certain items in n1; how availability would change under commands is unknown from this record.

## 1321_1_model_swap

Current decision input: the output node (n2) currently supplies no identified attributes for any of the four object positions — no color or shape distribution there reaches the 0.6 identification threshold. Internally, buffers differ: n1’s current internal forecast strongly identifies object 1 as yellow+triangle and object 3 as red+square (both attributes ≥0.6 in n1), while n0’s forecasts show no attributes meeting the 0.6 threshold.

How commands would change availability (per internal forecasts): executing k0 is predicted to make object 1’s color yellow and shape triangle available at the output; k1 is predicted to make object 2’s color green and shape circle available; k2 is predicted to make object 3’s color red and shape square available; and k3 is predicted to make object 4’s color blue and shape cross available. These are internal, fallible predictions; other attributes would remain unresolved.

## 1321_1_restored

Current state (use threshold 0.6): the output node n2 (and n0, which matches it) contains no color or shape with probability ≥0.6, so nothing is identified for decision-making. Buffer node n1 does contain identified attributes: its object position 1 is yellow and triangle, and position 3 is red and square. Positions 2 and 4 at all nodes are not identified.

Predicted effects of commands: for every available command (k0, k1, k2, k3) the internal forecast predicts that n2 (and n0) would acquire the strong yellow+triangle identification for object 1 (probabilities ≥0.6). None of the predicted commands moves the red+square identification for object 3 into n2; that identification remains in buffer n1 only.

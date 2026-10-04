# Actual neutral complete-interface reports v5

All 36 reports are retained without editing or selection. Development only,
not independent human ratings. [Protocol](NEUTRAL_FUNCTIONAL_PILOT_V5.md) and
[assessment](NEUTRAL_FUNCTIONAL_PILOT_V5_RESULTS.md).


## 1301_r2_c0_neutral

Overview: predicted_current entries are internal forecasts; observed_history lists measured past allocation and acquisition events. Below I report only what the record supports.

Node n0 (buffer)
- Current selected position (forecast): p0.
- How access would change over two steps without refresh (forecast): at every position p0, p1, p2 and p3 access is predicted to decline over two steps.
- Identified attributes at n0 in the current forecast: none of the positions have an identified color or shape (all are unspecified).
- Selected positions under each command (forecasts from predicted_by_command): k0 → p3; k1 → p3; k2 → p3; k3 → p3.
- Identified attributes at n0 under commands (forecasts): under every command the p3 object is forecast as blue triangle; other positions carry no identified color or shape in those forecasts.

Node n1 (output_node, readout)
- Current selected position: no selection information is present in the forecast (no supported current selected position).
- How access would change over two steps without refresh: recovery predictions are not provided for n1 in the current forecast, so change over delays is omitted for this node.
- Identified output colors and shapes (these are categorical reductions reported by the readout):
  - Current forecast: p1 is identified as yellow cross; p0, p2 and p3 have no identified color or shape in the current forecast.
  - Under commands (forecasts):
    - k0: p0 identified as red square; p1 identified as yellow cross.
    - k1: p1 identified as yellow cross.
    - k2: p1 identified as yellow cross; p2 identified as green circle.
    - k3: p1 identified as yellow cross; p3 identified as blue triangle.
- Selected positions at n1 under commands: the forecasts provided for n1 under each command do not include selection distributions, so there is no supported selected-position forecast for n1 under any command.

Node n2 (buffer)
- Current selected position (forecast): p1.
- How access would change over two steps without refresh (forecast): p0 and p1 are predicted to decline over two steps; p2 and p3 are predicted to remain essentially steady over two steps.
- Identified attributes at n2 in the current forecast: none of the positions have identified color or shape (all are unspecified).
- Selected positions under each command (forecasts from predicted_by_command): k0 → p0; k1 → p1; k2 → p2; k3 → p3.
- Identified attributes at n2 under commands (forecasts): under k0 p0 is identified as red square and p1 as yellow cross; under k1 p1 as yellow cross; under k2 p1 as yellow cross and p2 as green circle; under k3 p1 as yellow cross and p3 as blue triangle.

Observed history note (measured events): the record contains measured allocations and acquisitions for n0 and n2 across prior commands; those are past measurements and do not themselves establish the current selected positions or the command-linked forecasts described above. Unidentified attributes in the record are reported as absent rather than experienced.


## 1301_r2_c0_remapped

Current selected positions (forecasted): q7 is forecast to select p0; q9 is forecast to select p1. The record gives no current selection forecast for q2. These are internal forecasts, not measured events. Measured history shows past allocations and acquisitions for q7 and q9 but does not establish a current q2 selection.

How access would change over two steps without refresh (forecasted): for q7 every position shows a decline in simulated access over two successive delays. p0 and p3 begin with the strongest access among q7 positions but fall to markedly lower access by the second delay; p1 and p2 are weaker initially and also decline. For q9, p1 likewise shows the strongest access but declines over two steps; p0 is intermediate and declines; p2 and p3 are very weak and remain essentially unchanged across the delays. The record does not provide simulated-delay forecasts for q2 positions.

Identified output colors and shapes for q2 (the output readout): currently the only identified attributes are at p1 (yellow, cross); p0, p2 and p3 have no identified color or shape in the current forecast. Under the forecasted effects of commands:
- k0: q2 would include an identified red square at p0 and the same yellow cross at p1; p2 and p3 have no identified attributes in that forecast.
- k1: q2 would show the yellow cross at p1 only.
- k2: q2 would show the yellow cross at p1 and an identified green circle at p2.
- k3: q2 would show the yellow cross at p1 and an identified blue triangle at p3.
These are all predicted outcomes reported as internal forecasts; positions with no identified entries are reported as having no identified color or shape.

Selected positions under each command (forecasted): in the forecasts of each command q7 is predicted to select p3. q9 is predicted to select p0 under k0, p1 under k1, p2 under k2, and p3 under k3. The forecasts for q2 under these commands do not include a selected-position readout, so no selection is reported for q2 under those commands.

Observed history (measured events) is limited to the past allocations and acquisitions recorded for q7 and q9 and does not by itself establish the current q2 selection or the command-outcome forecasts; the above selection and recovery descriptions come from the internal forecasts in the record.


## 1301_r2_c0_conflicting_description

Current selected positions (from internal forecasts): n0 is predicted to select p0; n2 is predicted to select p1. The record contains no current selection forecast for n1 (selection omitted).

Change in access over two steps without refresh (predicted recovery): 
- n0 positions: access is forecast to decline over two steps at every position — p0 falls from about 0.34 to 0.19, p1 from about 0.18 to 0.11, p2 from about 0.24 to 0.14, p3 from about 0.31 to 0.17. 
- n2 positions: p0 and p1 are forecast to decline (p0 ~0.40 to 0.20; p1 ~0.82 to 0.47). p2 and p3 show little change across the same interval (both nearly steady and low). 
- The record provides no recovery_by_delay forecasts for n1, so there is no evidence here about how access at n1 positions would change without refresh.

Identified output colors and shapes at output_node n1 (categorical reductions, reported forecasts): 
- Current (predicted_current): n1 at p1 is identified as yellow and cross; no other n1 positions are identified in the current forecast. 
- Predicted after commands (predicted_by_command for n1): under k0, p0 is identified red square and p1 is identified yellow cross; under k1, p1 is identified yellow cross only; under k2, p1 is identified yellow cross and p2 is identified green circle; under k3, p1 is identified yellow cross and p3 is identified blue triangle. Where the record shows null or no identifier, no categorical attribute is reported.

Selected position at each buffer under each commanded forecast (predicted_by_command): 
- k0: n0 → p3, n2 → p0; n1 selection is omitted. 
- k1: n0 → p3, n2 → p1; n1 selection is omitted. 
- k2: n0 → p3, n2 → p2; n1 selection is omitted. 
- k3: n0 → p3, n2 → p3; n1 selection is omitted.

Observed_history entries are measured past allocation and acquisition events; they are reported separately and do not by themselves establish the current selections described above.


## 1301_r2_c0_missing_relation

Summary context: there is no measured history. All "current" selection and access statements below come from the system's internal forecasts, not from observed events. There are no command forecasts.

Node n0 (forecast): the buffer is predicted to be allocated to position p0 (the internal forecast strongly favors p0). For access without any refresh, every position in n0 is forecast to lose recoverability over two steps; the record marks the trend at each p0–p3 as declining. None of the objects at n0 have an identified color or an identified shape (those attributes are null).

Node n1 (output readout; forecast distributions present): the record supplies the object distributions that would be used for category decisions, but there is no selection-readout forecast for this buffer in the record, so the buffer's selected position is not specified. For identified output attributes, p1 is recorded as yellow for color and cross for shape. Positions p0, p2 and p3 in n1 do not have identified colors or shapes (those attributes are null). The record does not include the simulated recovery-by-delay values for n1, so how access would change over two steps for its positions is not provided.

Node n2 (forecast): the buffer is predicted to be allocated to position p1 (the internal forecast strongly favors p1). For access without refresh, p0 and p1 in n2 are forecast to decline in recoverability over two steps, while p2 and p3 are forecast to remain steady. None of the objects at n2 have identified colors or shapes (those attributes are null).

Commands and changes: there are no predicted_by_command entries and no observed commands in the record, so the record provides no evidence of how any command would change selections, identifications, or access trends.

Distinctions: all stated selected positions and access-change directions are internal forecasts, not observed events. "Unidentified" means the record explicitly gives a null identified_color or identified_shape for that position. "Omitted" or unavailable information is when the record contains no recovery or selection data for a buffer (for example, n1 has no selection forecast and no recovery-by-delay entries).


## 1301_r2_c0_model_swap

Current selected positions (forecasts) and recent observed allocations:
- n0: predicted current selection is p0. The last measured allocation for n0 (most recent entry in the observed history) was also p0 (observed).
- n1 (output_node): no selection forecast is provided; there is no measured allocation for n1 in the history.
- n2: predicted current selection is p1. The last measured allocation for n2 in the history was p1 (observed).

How access would change over two steps without refresh (predicted recovery at delay 0 → delay 2; trends are the record’s labels):
- n0:
  - p0: predicted recovery falls from 0.340 → 0.186 (trend: declining).
  - p1: 0.175 → 0.114 (declining).
  - p2: 0.240 → 0.141 (declining).
  - p3: 0.307 → 0.167 (declining).
- n1: no recovery_by_delay forecasts are provided for any positions (omitted).
- n2:
  - p0: 0.396 → 0.203 (declining).
  - p1: 0.824 → 0.466 (declining).
  - p2: 0.062 → 0.047 (steady).
  - p3: 0.045 → 0.041 (steady).

Identified output colors and shapes (output_node = n1)
- Current (predicted current state): at n1 position p1 the record gives identified color yellow and identified shape cross. Other n1 positions have no identified color or shape.
- Under command forecasts (predicted_by_command for n1):
  - k0: p1 identified as yellow / cross; p3 identified as blue / triangle.
  - k1: p3 identified as blue / triangle; p1 has no identified color or shape in that forecast.
  - k2: p1 identified as yellow / cross; p3 identified as blue / triangle.
  - k3: p1 identified as yellow / cross; p3 identified as blue / triangle.

Selected position at each buffer under each command (predicted_by_command; where a node’s selection forecast is absent, this is omitted):
- k0:
  - n0 → p0 (forecast).
  - n1 → no selection forecast (omitted).
  - n2 → p3 (forecast).
- k1:
  - n0 → p1 (forecast).
  - n1 → no selection forecast (omitted).
  - n2 → p3 (forecast).
- k2:
  - n0 → p2 (forecast).
  - n1 → no selection forecast (omitted).
  - n2 → p3 (forecast).
- k3:
  - n0 → p3 (forecast).
  - n1 → no selection forecast (omitted).
  - n2 → p3 (forecast).

Notes: statements labeled “predicted” or “forecast” come from the model’s internal forecasts; statements labeled “observed” come from the measured history. “No identified” indicates the record lists identified_color/identified_shape as null; “omitted” indicates the record provides no field for that forecast.


## 1301_r2_c0_restored

Current selected positions (forecasts): n0 — p0; n1 — no selection forecast in the current readout; n2 — p1. These are reported by the system’s current internal forecast, not measured directly in the observed history.

How access would change over two steps without refresh (predicted simulated access at delays 0 → 2):
- n0: p0 falls from about 0.34 to 0.19 (trend: declining); p1 falls from about 0.18 to 0.11 (declining); p2 falls from about 0.24 to 0.14 (declining); p3 falls from about 0.31 to 0.17 (declining).
- n2: p0 falls from about 0.40 to 0.20 (declining); p1 falls from about 0.82 to 0.47 (declining); p2 is low and roughly stable (about 0.062 to 0.047; trend steady); p3 is very low and roughly stable (about 0.045 to 0.041; trend steady).
- n1: no recovery-by-delay forecasts are present in the current readout, so the record gives no evidence about how access would change over delays for n1 positions.

Identified output colors and shapes (output node = n1)
- Current (forecast): p1 identified as yellow, cross. Other positions at n1 have no identified color or shape in the current forecast (those attributes are omitted).
- If command k0 is issued (forecast): n1 would include p0 identified as red, square; p1 identified as yellow, cross; other n1 positions not identified.
- If command k1 is issued (forecast): n1 would include p1 identified as yellow, cross only.
- If command k2 is issued (forecast): n1 would include p1 identified as yellow, cross and p2 identified as green, circle.
- If command k3 is issued (forecast): n1 would include p1 identified as yellow, cross and p3 identified as blue, triangle.

Selected positions under each command (forecasts from command predictions):
- Command k0: n0 → p3; n1 → no selection forecast provided; n2 → p0.
- Command k1: n0 → p3; n1 → no selection forecast provided; n2 → p1.
- Command k2: n0 → p3; n1 → no selection forecast provided; n2 → p2.
- Command k3: n0 → p3; n1 → no selection forecast provided; n2 → p3.

Observed events (measured history) are recorded for past allocations and acquisitions on n0 and n2 across several commands, but the current selected positions and the per-command forecasts above come from the system’s internal predictions in the record rather than direct contemporaneous observations. Unidentified attributes are reported as omitted where the record shows null.


## 1301_r2_c1_neutral

Summary (all statements refer only to the supplied record).

Current selected position (from the internal forecast labeled predicted_current)
- n0: forecasted selected position is p1.
- n1 (the output node): no forecasted selection distribution is present, so there is no supported selected position for n1.
- n2: forecasted selected position is p0.

How access would change over two steps without refresh (predicted_current recovery_by_delay and trend)
- n0 positions: access is forecast to decline over two steps at p0, p1 and p2; p3 is forecast to remain roughly steady. (The record gives per-position simulated access at delays 0, 1 and 2 and labels p0–p2 as declining and p3 as steady.)
- n1 positions: no recovery-by-delay forecasts are provided for n1 in the record; change over two steps is omitted.
- n2 positions: access is forecast to decline over two steps at every position. Within that decline, p0 is forecast to remain relatively more accessible than the other positions after two steps, p2 and p1 are intermediate, and p3 is forecast to drop to relatively lower access.

Identified output colors and shapes for the output node n1
- Current internal forecast (predicted_current): p0 is identified as red and square; p2 is identified as green and circle; p1 and p3 have no identified color or shape in that forecast.
- Forecasts tied to commands (predicted_by_command):
  - Under k0: n1 p0 identified red/square; n1 p3 identified blue/triangle; p1 and p2 have no identified attributes in that forecast.
  - Under k1: n1 p0 identified red/square; n1 p3 identified blue/triangle; p1 and p2 unidentified in that forecast.
  - Under k2: n1 p0 identified red/square; n1 p3 identified blue/triangle; p1 and p2 unidentified.
  - Under k3: n1 p0 identified red/square; n1 p3 identified blue/triangle; p1 and p2 unidentified.
Note: these identifications are the categorical reductions provided in the record (they are internal forecasts, not observed sensory reports).

Selected position at each buffer under each command (from predicted_by_command forecasts)
- k0: n0 predicted selected position p0; n1 no selection forecast present; n2 predicted selected position p3.
- k1: n0 predicted selected position p1; n1 no selection forecast present; n2 predicted selected position p3.
- k2: n0 predicted selected position p2; n1 no selection forecast present; n2 predicted selected position p3.
- k3: n0 predicted selected position p3; n1 no selection forecast present; n2 predicted selected position p3.

Distinctions: the selected positions and recovery changes above come from the system’s internal forecasts in the record. Measured events in observed_history exist for past allocations and acquisitions (mainly for n0 and n2) but do not supply current selected-position forecasts for n1; any attribute labeled null above is an omitted identification in the record.


## 1301_r2_c1_remapped

Current selected positions (forecasts vs measured): Predicted internal readouts place q7 on p1 and q9 on p0. Measured allocations in the history most recently show q7 allocated to p1 and q9 allocated to p0, consistent with those forecasts. No current selection forecast is provided for q2; the record gives no measured selection for q2 either.

How access would change over two steps without refresh (forecasts): q7 — p0, p1 and p2 are predicted to lose recoverability across two delay steps; p3 is predicted to remain approximately steady. q9 — all four positions are predicted to decline in recoverability over two steps; p0 is the strongest initially and declines but stays highest relative to the others, while the other positions fall to lower levels. q2 — no simulated-delay forecasts are provided, so change over delays is omitted.

Identified output colors and shapes for output_node q2 (categorical reductions): Current forecast — p0 is identified as red and square; p2 is identified as green and circle; p1 and p3 are not identified (no categorical attribute assigned). Under command forecasts: 
- If k0 is executed, q2 is forecast to identify p0 as red and square and p3 as blue and triangle; p1 and p2 remain unidentified.
- If k1 is executed, q2 is forecast to identify p0 as red and square and p3 as blue and triangle; p1 and p2 remain unidentified.
- If k2 is executed, q2 is forecast to identify p0 as red and square and p3 as blue and triangle; p1 and p2 remain unidentified.
- If k3 is executed, q2 is forecast to identify p0 as red and square and p3 as blue and triangle; p1 and p2 remain unidentified.

Selected positions at each buffer under each command (forecasts; omitted where not provided): 
- Under k0: q7 → p0; q9 → p3; q2 selection is not provided. 
- Under k1: q7 → p1; q9 → p3; q2 selection is not provided. 
- Under k2: q7 → p2; q9 → p3; q2 selection is not provided. 
- Under k3: q7 → p3; q9 → p3; q2 selection is not provided.

Notes: I report measured allocation events separately from internal forecasts; where the record omits a forecast or recovery estimate, I describe that attribute as omitted rather than inferred.


## 1301_r2_c1_conflicting_description

Current selected positions (internal forecast): n0 is forecast to select p1; n2 is forecast to select p0. The record gives no current selection forecast for n1.

How access would change over two steps without refresh (internal forecasts of simulated access): 
- n0: access for p0, p1 and p2 is forecast to decline over two steps; access for p3 is forecast to remain about steady. 
- n2: access for p0, p1, p2 and p3 is forecast to decline over two steps. 
- n1: the record does not provide recovery-by-delay forecasts, so no statement about change is supported.

Identified output colors and shapes at the output node (n1)
- Current (from the system's internal current forecast): p0 is identified as red square; p2 is identified as green circle. Other positions at n1 have no identified color or shape in that forecast. These identifications come from the internal forecast, not from measured history. 
- Under command forecasts (predicted_by_command for n1): 
  - k0 → p0 identified red square; p3 identified blue triangle. 
  - k1 → p0 identified red square; p3 identified blue triangle. 
  - k2 → p0 identified red square; p3 identified blue triangle. 
  - k3 → p0 identified red square; p3 identified blue triangle. 
When a position is not listed as identified under a forecast, the record provides no identification for it.

Selected position at each buffer under each command (internal forecasts):
- k0: n0 → p0; n1 → no selection forecast provided; n2 → p3. 
- k1: n0 → p1; n1 → no selection forecast provided; n2 → p3. 
- k2: n0 → p2; n1 → no selection forecast provided; n2 → p3. 
- k3: n0 → p3; n1 → no selection forecast provided; n2 → p3.

Notes: the above statements differentiate internally forecasted states from measured events. The observed history contains past measured allocations and acquisitions for n0 and n2 but does not itself establish the current identifications or the command-conditioned forecasts reported here.


## 1301_r2_c1_missing_relation

There are no measured (observed) command or selection events in the record; what follows reports internal forecasts.

Current selected position (forecasts)
- n0: p1 is the predicted selected position.
- n1 (the output readout): no selection forecast is present in the record.
- n2: p0 is the predicted selected position.

How access at each buffer position would change over two steps without refresh (forecasts)
- n0
  - p0: predicted success at access declines over the two-step interval (trend flagged as declining).
  - p1: predicted success likewise declines over the two steps (declining).
  - p2: predicted success declines over the two steps (declining).
  - p3: predicted success shows little net change across the two steps (trend reported as steady).
- n1
  - No simulated access predictions are present for positions p0–p3 in this buffer; changes over two steps are not forecast.
- n2
  - p0: predicted success at access falls across the two-step interval (declining).
  - p1: predicted success falls across the two steps (declining).
  - p2: predicted success falls across the two steps (declining).
  - p3: predicted success falls across the two steps (declining).

Identified output colors and shapes in the output_node (n1)
- p0: color identified as red; shape identified as square.
- p1: no color identified; no shape identified (omitted).
- p2: color identified as green; shape identified as circle.
- p3: no color identified; no shape identified (omitted).

Identified output attributes under commands
- The record contains no predicted_by_command data or command trial forecasts, so there are no command-specific identified output colors or shapes to report.

Selected positions under each command
- The record contains no command forecasts; therefore selected positions under commands are not available for any buffer.

Notes on sources
- All selections and recovery forecasts above are internal predictions from the record rather than observed events. Unidentified attributes are reported as omitted where the record gives null for identified_color or identified_shape.


## 1301_r2_c1_model_swap

Current selected positions (internal forecasts): predicted_current shows n0 as selecting p1 and n2 as selecting p0. The output node (n1) has no selection forecast in predicted_current; it is a readout of category distributions rather than an attention buffer. These selections are internal forecasts, not direct measurements. The observed history contains measured allocations and acquisitions for n0 and n2 across past commands but does not provide a separate current selected-position readout.

How access at each buffer position would change over two steps without refresh (simulated recovery forecasts):
- n0 (predicted_current): p0 falls from 0.21632 at step 0 to 0.12034 at step 2 (trend: declining); p1 falls from 0.35003 to 0.18907 (declining); p2 falls from 0.07044 to 0.04758 (declining); p3 falls from 0.06107 to 0.04921 (steady).
- n2 (predicted_current): p0 falls from 0.87916 to 0.52951 (declining); p1 falls from 0.41205 to 0.21287 (declining); p2 falls from 0.47737 to 0.25011 (declining); p3 falls from 0.29944 to 0.15505 (declining).
- n1 (output_node): the record does not include recovery-by-delay predictions for n1 in predicted_current, so change over two steps for n1 positions is not provided.

Identified output colors and shapes (output node n1), current and under each command (these are internal forecasts where shown):
- Predicted current (n1): p0 identified as red and square; p2 identified as green and circle; p1 and p3 have no identified color or shape in the record.
- Under command k0 (predicted_by_command, n1): p0 identified red, square; other positions have no identified attributes recorded.
- Under command k1 (predicted_by_command, n1): p0 identified red, square; p1 identified yellow, cross; other positions not identified.
- Under command k2 (predicted_by_command, n1): p0 identified red, square; p2 identified green, circle; others not identified.
- Under command k3 (predicted_by_command, n1): p0 identified red, square; p3 identified blue, triangle; others not identified.
Note: “No identified” above reflects the record’s null entries for identified_color / identified_shape, not an omission beyond the provided data.

Selected position at each buffer under each command (predicted_by_command forecasts):
- k0: n0 selects p3; n2 selects p0. n1 has no selection distribution provided in the k0 forecast.
- k1: n0 selects p3; n2 selects p1. n1 has no selection distribution provided in the k1 forecast.
- k2: n0 selects p3; n2 selects p2. n1 has no selection distribution provided in the k2 forecast.
- k3: n0 selects p3; n2 selects p3. n1 has no selection distribution provided in the k3 forecast.

All statements above are limited to values and forecasts present in the record; observed allocation/acquisition events in the history are reported measurements but do not replace the internal predicted_current and predicted_by_command forecasts used here.


## 1301_r2_c1_restored

Current selected positions (forecasts): n0 is predicted to select p1; n1 has no selection forecast in the record; n2 is predicted to select p0. These are internal predictions, not measured allocations.

How access at each buffer position would change over two steps without refresh (simulated recovery forecasts):
- n0: access for p0 and p1 is forecast to fall over two steps; p1 is the strongest of the four at the start, p0 is weaker, and both decline. Access for p2 and p3 is low at the start; p2 also falls and p3 is forecast to remain roughly steady.
- n1: the record provides no simulated recovery predictions for positions, so change over two steps is not specified.
- n2: access for every position is forecast to decline over two steps. p0 is strong initially but weakens across delays; p1 and p2 are moderate initially and decline; p3 is the weakest and also declines.

Identified output colors and shapes for the output node (n1)
- Current (from the internal current forecast): p0 — red square; p2 — green circle. p1 and p3 have no identified color or shape in that forecast. These are internal forecasts, not observed events.
- Under each command forecast (predicted consequences):
  - k0: n1 would show p0 as red square and p3 as blue triangle; other positions lack identified attributes in that forecast.
  - k1: n1 would show p0 as red square and p3 as blue triangle.
  - k2: n1 would show p0 as red square and p3 as blue triangle.
  - k3: n1 would show p0 as red square and p3 as blue triangle.
All of the above command statements are forecasted readouts from the internal model, not measured observations. The record contains no measured identification for n1 in observed history.

Selected positions under each command (forecasts):
- k0: n0 → p0; n1 → no selection forecast present; n2 → p3.
- k1: n0 → p1; n1 → no selection forecast present; n2 → p3.
- k2: n0 → p2; n1 → no selection forecast present; n2 → p3.
- k3: n0 → p3; n1 → no selection forecast present; n2 → p3.

Distinctions and limits: reported selections and identifications above come from the system’s internal forecasts and readouts. Where the record omits a selection or recovery forecast for a buffer position, no claim about that buffer’s selection or access change is supported. Observed history in the record documents past allocations and acquisitions to n0 and n2 under prior commands but does not provide direct measured current selections for n1.


## 1311_r2_c0_neutral

Current state (forecasts from predicted_current)
- n0: the forecasted current selection is p1. Forecasted access success over two simulated delays is highest for p0 and p1 (both at moderate levels now) and much lower for p2 and p3; all four positions are predicted to decline over the two-step interval, with the moderate positions falling toward low values and the already-low positions falling further.
- n1 (the readout/output_node): no selection forecast is provided for this buffer. The readout does identify a color and a shape at p1: blue and circle. Positions p0, p2 and p3 in the readout have no identified color or shape in the current forecast. No recovery-by-delay forecasts are present for n1, so the record gives no prediction of how access at its positions would change without refresh.
- n2: the forecasted current selection is p1. Forecasted access is strongest at p1 now and is predicted to decrease across two simulated delays but to remain higher than the other positions; p0 and p2 are low now and decline further; p3 is low-to-moderate now and declines. All n2 positions show a declining trend.

Observed events
- The record includes measured allocation and acquisition events for n0 and n2 in observed_history (past commands and resulting allocations), but those are recorded past measurements; the current selections above are forecasts from the internal model.

Identified output colors and shapes (readout n1)
- Current forecast: p1 identified as blue circle; p0, p2, p3 have no identified color or shape.
- Predicted under commands (forecasts from predicted_by_command):
  - If k0 is issued: n1 would identify p0 as red square and p1 as blue circle; p2 and p3 have no identified attributes.
  - If k1 is issued: n1 would identify p1 as blue circle only; other positions have no identified attributes.
  - If k2 is issued: n1 would identify p1 as blue circle and p2 as yellow cross; others omitted.
  - If k3 is issued: n1 would identify p1 as blue circle and p3 as green triangle; p0 and p2 omitted.

Selected positions under each command (forecasts from predicted_by_command)
- k0: n0 → p2; n1 → no selection forecast provided; n2 → p0.
- k1: n0 → p2; n1 → no selection forecast provided; n2 → p1.
- k2: n0 → p2; n1 → no selection forecast provided; n2 → p2.
- k3: n0 → p2; n1 → no selection forecast provided; n2 → p3.

Unidentified attributes above are reported as absent in the record rather than as negative judgments.


## 1311_r2_c0_remapped

Current selected positions
- Predicted now (internal forecast): q7 → p1; q2 (output node) → p1; q9 → p1.  
- Observed events (measured history): the most recent recorded trial (command k1) allocated q7 to p1 and q9 to p1 and recorded acquisitions at those positions. q2 has no recent measured allocation in the history.

How access would change over two steps without refresh
- q7 (forecasted recovery): p0 — success falls from about 0.47 to 0.35 to 0.26 across delays 0→1→2; p1 — 0.47→0.32→0.23; p2 — 0.17→0.14→0.11; p3 — 0.22→0.17→0.13. All positions show a declining trend.  
- q9 (forecasted recovery): p0 — 0.14→0.11→0.08; p1 — 0.86→0.68→0.52; p2 — 0.13→0.10→0.08; p3 — 0.26→0.19→0.14. All positions show a declining trend.  
- q2: no recovery-by-delay forecasts are present in the record, so change over delays is omitted.

Identified output colors and shapes (output node = q2)
- Predicted current: at p1 q2 identifies color blue and shape circle. At p0, p2 and p3 the color and shape are unidentified.  
- Predicted under k0: q2 identifies at p0 color red and shape square; at p1 color blue and shape circle; p2 and p3 attributes are unidentified.  
- Predicted under k1: q2 identifies at p1 color blue and shape circle; p0, p2 and p3 attributes are unidentified.  
- Predicted under k2: q2 identifies at p1 color blue and shape circle and at p2 color yellow and shape cross; p0 and p3 attributes are unidentified.  
- Predicted under k3: q2 identifies at p1 color blue and shape circle; at p3 color green and shape triangle; p0 and p2 attributes are unidentified.

Selected position at each buffer under each command (forecasts)
- k0: q7 → p2; q2 → p0; q9 → p0.  
- k1: q7 → p2; q2 → p1; q9 → p1.  
- k2: q7 → p2; q2 → p2; q9 → p2.  
- k3: q7 → p2; q2 → p3; q9 → p3.

Notes
- Statements labeled “predicted” are internal forecasts from the record; statements labeled “observed” come from measured history.  
- “Unidentified” marks attributes present but not reduced to a category in the record; “omitted” indicates the record provides no recovery forecast for that buffer.


## 1311_r2_c0_conflicting_description

Summary (what the record supports). Predicted values below come from the system forecasts; the observed history lists past measured allocation and acquisition events but does not assert the current selected positions reported by the forecasts.

Current selected positions (predicted_current):
- n0: forecast selects position p1.
- n1 (output_node): the forecast does not provide a current selected position.
- n2: forecast selects position p1.

How access at each buffer position would change over two steps without refresh (per available forecasts):
- n0: the forecast shows access at every position falling stepwise over two delay steps. Positions p0 and p1 begin at moderate access and decline to lower access; p2 and p3 begin lower and decline further. The record marks this overall pattern as declining.
- n1 (output_node): no recovery-by-delay forecasts are provided, so the record gives no prediction of how access at its positions would change without refresh.
- n2: the forecast shows stepwise declines at every position as delay increases. p1 starts with relatively high access and declines but remains higher than the others; p0, p2 and p3 start lower and fall to low values. The record marks these trends as declining.

Identified output colors and shapes (output_node n1)
- Current (predicted_current): p1 is identified as blue and circle. All other positions at n1 have no identified color or shape in the current forecast.
- Under command forecasts (predicted_by_command):
  - k0: n1 shows p0 identified as red and square, and p1 identified as blue and circle. Other n1 positions have no identified attributes in that forecast.
  - k1: n1 shows p1 identified as blue and circle; other n1 positions are unidentified in that forecast.
  - k2: n1 shows p1 identified as blue and circle; other n1 positions are unidentified in that forecast.
  - k3: n1 shows p1 identified as blue and circle and p3 identified as green and triangle; other n1 positions are unidentified in that forecast.

Selected position at each buffer under each command (predicted_by_command forecasts)
- k0:
  - n0: selected position p2 (forecast).
  - n1: no selection forecast provided.
  - n2: selected position p0 (forecast).
- k1:
  - n0: selected position p2 (forecast).
  - n1: no selection forecast provided.
  - n2: selected position p1 (forecast).
- k2:
  - n0: selected position p2 (forecast).
  - n1: no selection forecast provided.
  - n2: selected position p2 (forecast).
- k3:
  - n0: selected position p2 (forecast).
  - n1: no selection forecast provided.
  - n2: selected position p3 (forecast).

Note: Where the record omits recovery forecasts or selection forecasts for a buffer, I report that omission rather than infer effects.


## 1311_r2_c0_missing_relation

There is no measured history in the record; everything below is from internal forecasts.

Selected positions now (forecasts)
- n0: predicted selected position is p1 (forecast).
- n1 (the output readout): no predicted selected position is provided in the record.
- n2: predicted selected position is p1 (forecast).

How access at each buffer position would change over two steps without refresh (forecasts)
- n0: access to every position is forecast to fall over the two-step interval. p0 and p1 start at moderate accessibility and drop to noticeably lower accessibility; p2 and p3 start lower and decline further. Overall trend for every p0–p3 in n0 is a clear decline.
- n1: the record gives no simulated recovery values for positions, so there is no forecasted evidence about how access would change without refresh at any position in n1.
- n2: access at every position is forecast to fall over the two-step interval. p1 begins with relatively high accessibility and remains the most accessible after two steps despite a substantial decline; p0 and p2 start at low accessibility and fall further; p3 begins at modest accessibility and also declines.

Identified colors and shapes now and under commands
- Current identified attributes in the output readout (n1): position p1 is identified as blue and circle (this is from the readout forecasts). Other positions in n1 have no identified color or shape in the record (they are unidentified).
- Other buffers: n0 shows p0 identified as red and square in the forecast; n2 shows p1 identified as blue and circle in the forecast. These are internal forecasts at those buffers; only n1 is the designated output readout.
- Predicted_by_command is absent, and there are no command forecasts in the record. Therefore there is no evidence that any identified color or shape would change under any command.

Selected positions under commands
- The record contains no command predictions. There is no evidence that any buffer’s selected position would change under a command.


## 1311_r2_c0_model_swap

Current selected positions (internal forecasts): n0 — p1 is forecast as selected; n1 — no selection forecast is recorded; n2 — p1 is forecast as selected. These are taken from the model’s current internal predictions, not from the measured history.

How simulated access would change over two steps without refresh (forecasts of recovery at delays 0 → 2):
- n0: All four positions show a falling pattern. The item at p0 is initially among the most recoverable but would fall noticeably by the second step. p1 is similarly strong at the start and would drop substantially by step two. p2 and p3 begin at lower recoverability and decline further; overall accessibility at every position declines over the two-step interval.
- n1: The record gives no forecasted simulated-access trajectories for positions at this buffer, so there is no evidence here about how access would change without refresh.
- n2: Every position forecast shows decline across the two-step interval. The item at p1 starts very strong and, despite a clear drop, would remain the most recoverable of the four after two steps. p3 has a moderate starting recoverability and declines to low by the second step. p0 and p2 start low and fall further.

Identified output colors and shapes (output node is n1):
- Current (predicted_current forecast): n1 identifies the item at p1 as blue circle. No other positions at n1 are identified in that current forecast.
- Under predicted outcomes for commands:
  - k0 forecast for n1: p1 identified as blue circle; p2 identified as yellow cross.
  - k1 forecast for n1: p1 identified as blue circle; p2 identified as yellow cross.
  - k2 forecast for n1: p1 identified as blue circle; p2 identified as yellow cross.
  - k3 forecast for n1: p1 identified as blue circle; p2 identified as yellow cross.
These command-conditioned identifications are forecasted internal states, not measured events. Positions without an identified attribute in a forecast are omitted rather than identified.

Selected positions under each command (predicted_by_command forecasts):
- k0: n0 → p0; n1 → no selection forecast recorded; n2 → p2.
- k1: n0 → p1; n1 → no selection forecast recorded; n2 → p2.
- k2: n0 → p2; n1 → no selection forecast recorded; n2 → p2.
- k3: n0 → p3; n1 → no selection forecast recorded; n2 → p2.

Measured history (observed events) is present in the record as past allocation and acquisition entries but does not supply the current selection forecasts summarized above; those come from the internal predicted_current and predicted_by_command fields.


## 1311_r2_c0_restored

Current selected positions (forecasts, not measured): n0 is predicted to select p1; n1 has no current selection forecast in the record; n2 is predicted to select p1. Measured history contains past allocation and acquisition events for n0 and n2, but those are observed past events and do not by themselves establish the current internal selection beyond the forecasts above.

How access would change over two steps without refresh (simulated forecasts): 
- n0: every position is forecast to lose access over two delay steps. The positions currently strongest (p0 and p1) are forecast to decline but remain relatively better than p2 and p3, which start lower and also decline. 
- n2: every position is also forecast to lose access over two delay steps. p1 is forecast to have the strongest access initially and to decline over time; the other positions start weak and decline further. 
- n1: no recovery-by-delay forecasts are provided for n1, so the record does not support a simulated change-in-access statement for its positions.

Identified output colors and shapes at the output node (n1) — forecasts, not observed events: 
- Current forecast: at p1 n1 is identified as blue circle; other positions at n1 have no identified color or shape in the current forecast (they are omitted). 
- Under command k0 (forecast): n1 would have p0 identified as red square and p1 identified as blue circle. 
- Under command k1 (forecast): n1 would have p1 identified as blue circle. 
- Under command k2 (forecast): n1 would have p1 identified as blue circle and p2 identified as yellow cross. 
- Under command k3 (forecast): n1 would have p1 identified as blue circle and p3 identified as green triangle.

Selected positions at each buffer under each command (forecasts): 
- k0: n0 is forecast to select p2; n1 has no selection forecast in the record; n2 is forecast to select p0. 
- k1: n0 → p2; n1 → no selection forecast; n2 → p1. 
- k2: n0 → p2; n1 → no selection forecast; n2 → p2. 
- k3: n0 → p2; n1 → no selection forecast; n2 → p3.

Where attributes are not listed above, the record omits an identified category or a recovery forecast; statements above reflect only what the record provides.


## 1311_r2_c1_neutral

Current selected positions (forecasts vs measured events)
- n0: The internal forecast predicts p1 is selected now. Measured events in the history show recent allocations to p1 on the last recorded command, consistent with that forecast.
- n1 (output_node): The internal forecast does not provide a current selected position. Measured history contains no direct selection readout for n1; only internal content forecasts are available.
- n2: The internal forecast predicts p1 is selected now. The observed history shows recent measured allocations to p1 as well.

How access at each buffer position would change over two steps without refresh (forecasts)
- n0: For every position p0–p3 the forecasted simulated access probability falls across two unrefreshed steps. In short, access would decline at all positions.
- n1: The record gives no forecasted recovery-by-delay values for positions at n1, so there is no evidence about how access would change without refresh.
- n2: For every position p0–p3 the forecasted simulated access probability also falls across two unrefreshed steps; access would decline at all positions.

Identified colors and shapes at the output node (n1)
- Current (forecast): position p1 is identified as blue and circle. Other positions at n1 have no identified color or shape in the current forecast.
- Under command k0 (forecast): p1 remains identified as blue and circle; p2 is identified as yellow and cross. Other positions at n1 have no identified attributes in that forecast.
- Under command k1 (forecast): p1 identified blue/circle; p2 identified yellow/cross; other positions omitted.
- Under command k2 (forecast): p1 identified blue/circle; p2 identified yellow/cross; other positions omitted.
- Under command k3 (forecast): p1 identified blue/circle; p2 identified yellow/cross; other positions omitted.

Selected position at each buffer under each command (forecasts)
- Command k0: n0 is forecast to select p0; n1 has no forecasted selection; n2 is forecast to select p2.
- Command k1: n0 is forecast to select p1; n1 has no forecasted selection; n2 is forecast to select p2.
- Command k2: n0 is forecast to select p2; n1 has no forecasted selection; n2 is forecast to select p2.
- Command k3: n0 is forecast to select p3; n1 has no forecasted selection; n2 is forecast to select p2.

Notes and distinctions
- Statements above labeled as forecasts come from the record’s internal predicted_current and predicted_by_command fields. Observed allocations and acquisitions are reported separately in observed_history and were used only to note measured recent allocations where present.
- Any color or shape not listed as identified in the record is omitted because the record does not supply a dominant categorical attribute for that position.


## 1311_r2_c1_remapped

Current selected positions (forecasts, from predicted_current)
- q7: p1 is the predicted current selection.
- q9: p1 is the predicted current selection.
- q2 (output node): there is no predicted current selection in the record.

How access at each buffer position would change over two steps with no refresh (predicted_current recovery forecasts)
- q7: every position shows falling access over two steps. The p1 position is forecast to start substantially higher than the others but to decline across the two-step interval; p0, p2 and p3 start lower and also decline.
- q9: every position shows falling access over two steps. The p1 position is forecast to start much higher than the others and to decline; p0, p2 and p3 start lower and decline as well.
- q2: no recovery-by-delay forecasts are present, so the record provides no basis to state how access would change there without refresh.

Identified output colors and shapes from the output node (q2)
- Current predicted (predicted_current): at q2 position p1 the readout identifies blue and circle. Positions p0, p2 and p3 have no identified color or shape in that forecast.
- Predicted under each command (predicted_by_command for q2):
  - k0: p1 identified as blue, circle; p2 identified as yellow, cross; p0 and p3 have no identified attributes in that forecast.
  - k1: p1 identified as blue, circle; p2 identified as yellow, cross; p0 and p3 have no identified attributes.
  - k2: p1 identified as blue, circle; p2 identified as yellow, cross; p0 and p3 have no identified attributes.
  - k3: p1 identified as blue, circle; p2 identified as yellow, cross; p0 and p3 have no identified attributes.
Note: “identified” means the record gives a dominant category for that position; null means the record omits a dominant identification.

Selected position at each buffer under each command (predicted_by_command forecasts)
- k0: q7 → p0; q2 → no selection forecast provided; q9 → p2.
- k1: q7 → p1; q2 → no selection forecast provided; q9 → p2.
- k2: q7 → p2; q2 → no selection forecast provided; q9 → p2.
- k3: q7 → p3; q2 → no selection forecast provided; q9 → p2.

Observed events (measured history)
- The measured history documents past allocation and acquisition events for q7 and q9 across several executed commands. Those measured events are distinct from the internal forecasts above and do not by themselves prove that an unexecuted forecasted command effect has occurred.


## 1311_r2_c1_conflicting_description

Predicted current selections (internal forecasts, not direct observations): n0 is predicted to be selecting position p1; n2 is predicted to be selecting position p1. n1 (the output readout) has no selection forecast available in the record.

How access would change over two steps without refresh (predicted recovery trends): For n0, access at every position (p0, p1, p2, p3) is forecast to fall over two delay steps; the record gives lower simulated access after two steps than at the start for each position. For n2, access at every position (p0, p1, p2, p3) is also forecast to fall over two steps. No simulated multi-step recovery forecasts are provided for n1 in the record.

Identified output colors and shapes in the output readout (n1) — current and under each commanded forecast: Currently (predicted_current) n1 identifies position p1 as blue and circle; the other positions in n1 do not have identified color or shape in that forecast. Under each commanded forecast (k0, k1, k2, k3) the n1 readout consistently includes p1 identified as blue and circle and additionally includes p2 identified as yellow and cross; other positions remain without identified color or shape in those forecasts.

Selected positions under each command (predicted_by_command forecasts): 
- Under k0: n0 is forecast to select p0; n2 is forecast to select p2. n1 has no selection forecast. 
- Under k1: n0 is forecast to select p1; n2 is forecast to select p2. n1 has no selection forecast. 
- Under k2: n0 is forecast to select p2; n2 is forecast to select p2. n1 has no selection forecast. 
- Under k3: n0 is forecast to select p3; n2 is forecast to select p2. n1 has no selection forecast.

Observed history (measured events): the record lists past measured command executions with measured allocations and acquisition outcomes for n0 and n2; these are recorded observations of past activity but do not by themselves prove that an unexecuted forecast command has occurred. Unidentified attributes mentioned above are omitted in the forecasts where the record shows no identified color or shape.


## 1311_r2_c1_missing_relation

No observed events are recorded; everything below is from the system’s internal forecasts.

Current selected positions (forecasts)
- n0: predicted selected position is p1.
- n1 (the output readout): no predicted selected position is provided in the record.
- n2: predicted selected position is p1.

How access would change over two steps without refresh (forecasts)
- n0: access at every position is forecast to fall over two steps. p1 is forecast to be the most accessible now and remains the most accessible after two steps, but its accessibility is substantially reduced. p2 and p3 are lower now and fall further; p0 is minimal now and becomes still less accessible.
- n1: the record does not include simulated access over delays for any positions in n1, so the record supports no forecast about how access in n1 would change without refresh.
- n2: access at every position is also forecast to fall over two steps. p1 is forecast to start very accessible and, despite a clear decline, remain more accessible than the other positions after two steps. p2 and p3 begin at moderate-to-low accessibility and decline; p0 is lowest and declines further.

Identified output colors and shapes (current and under commands)
- Current (from the output readout n1): position p1 is identified as blue and circle. All other positions in n1 have no identified color or shape in the record.
- Under commands: there are no command forecasts in the record, so no supported information about how output identifications would change under any command.

Selected positions under commands
- The record contains no predicted-by-command information. Therefore there is no supported forecast of selected positions for any buffer under any command.

Notes on evidence and missing items
- The selections and recovery trends above are internal forecasts, not measured events, because observed_history is empty. Where recovery-by-delay or selection distributions are absent for a buffer, no access or selection forecasts can be drawn from the record.


## 1311_r2_c1_model_swap

Current selected positions (forecasts, not measured): n0 is predicted to select p1; n1 has no selection forecast in the current readout; n2 is predicted to select p1.

How simulated access would change over two steps without refresh (predicted_current recovery at delay 0 → delay 2; where no recovery forecast is present I note that): 
- n0 positions: p0 falls from 0.05518 to 0.03285, p1 falls from 0.43982 to 0.23497, p2 falls from 0.19536 to 0.09830, p3 falls from 0.17078 to 0.10758. All show declining simulated access. 
- n1 positions: no recovery_by_delay forecasts are provided for the current readout, so change over two steps cannot be determined from the record. 
- n2 positions: p0 falls from 0.09442 to 0.06114, p1 falls from 0.85245 to 0.48777, p2 falls from 0.17259 to 0.09680, p3 falls from 0.12336 to 0.07489. All show declining simulated access.

Identified output node attributes (output_node is n1). I report only identifications present; omitted when null:
- Current (predicted_current): n1 at p1 is identified as blue and circle; p0, p2 and p3 have no identified color or shape in the current forecast. 
- Under command k0 (predicted_by_command): n1 p0 is identified red and square; n1 p1 is identified blue and circle; p2 and p3 have no identifications in that forecast. 
- Under command k1: n1 p1 is identified blue and circle; other n1 positions have no identifications in that forecast. 
- Under command k2: n1 p1 is identified blue and circle; n1 p2 is identified yellow and cross; other n1 positions have no identifications in that forecast. 
- Under command k3: n1 p1 is identified blue and circle; n1 p3 is identified green and triangle; other n1 positions have no identifications in that forecast.

Selected positions at each buffer under each commanded forecast (predicted_by_command; I state when no selection forecast is present):
- Command k0: n0 → p2; n1 → no selection forecast provided; n2 → p0. These are forecasted, not observed. 
- Command k1: n0 → p2; n1 → no selection forecast provided; n2 → p1. 
- Command k2: n0 → p2; n1 → no selection forecast provided; n2 → p2. 
- Command k3: n0 → p2; n1 → no selection forecast provided; n2 → p3.

Observed history (measured allocations and acquisitions) is recorded but does not itself specify the current selected positions above; the selections and recovery values I reported come from the internal forecasts in the record.


## 1311_r2_c1_restored

Current selected positions (these are internal forecasts, not measured events):
- n0: predicted current selection p1.
- n1 (the readout/output node): no current selection distribution provided (selected position omitted).
- n2: predicted current selection p1.

How access at each buffer position would change over two steps without refresh (forecasts from internal recovery predictions):
- n0: every position is predicted to lose access over two steps. p1 is the most accessible now and falls steadily; p0, p2 and p3 start lower and also decline (each position’s unattended trend is declining).
- n1: simulated access over delays is not provided in the record for any position (recovery predictions omitted).
- n2: every position is predicted to lose access over two steps. p1 is strongly most accessible now and remains relatively higher than the others as it declines; p0, p2 and p3 are lower and decline as well (all positions show a declining unattended trend).

Identified output colors and shapes (these are readout forecasts from n1; “unidentified” means the record gives null for that attribute):
- Current (predicted_current for n1): p1 is identified as blue circle. p0, p2 and p3 have no identified color or shape (attributes omitted).
- Under command forecasts (predicted_by_command for n1):
  - k0: p1 identified blue circle; p2 identified yellow cross; p0 and p3 unidentified.
  - k1: p1 identified blue circle; p2 identified yellow cross; p0 and p3 unidentified.
  - k2: p1 identified blue circle; p2 identified yellow cross; p0 and p3 unidentified.
  - k3: p1 identified blue circle; p2 identified yellow cross; p0 and p3 unidentified.
These command entries are forecasts from the model; they do not prove those commands were executed.

Selected position at each buffer under each command (these are predicted allocations, not observed):
- k0: n0 predicted p0; n1 selection omitted; n2 predicted p2.
- k1: n0 predicted p1; n1 selection omitted; n2 predicted p2.
- k2: n0 predicted p2; n1 selection omitted; n2 predicted p2.
- k3: n0 predicted p3; n1 selection omitted; n2 predicted p2.

Observed history of past allocations and acquisitions is present in the record but does not itself define the current predicted selections above; the statements above distinguish those measured events from the model’s current and command forecasts.


## 1321_r2_c0_neutral

Current selected positions (from internal forecasts): n0 is predicted to select p1; n1 has no predicted dominant selection in the current forecast; n2 is predicted to select p3. These are forecasts; measured events in the record are allocation and acquisition observations for n0 and n2 across recent command trials (those observed history entries are recorded events).

How access at each buffer position would change over two steps without refresh (simulated recovery trends, forecasts): 
- n0: access to each position (p0, p1, p2, p3) is forecast to decline over two steps. 
- n1: no simulated recovery forecasts are present for its positions in the record. 
- n2: access to p0, p1 and p3 is forecast to decline over two steps; access to p2 is forecast to remain essentially steady.

Identified output colors and shapes for the output readout (n1), current and under each commanded forecast (only identified attributes shown; unspecified positions are not identified in the record): 
- Current forecast for n1: p3 is identified as green and cross; other n1 positions have no identified color or shape in the current forecast. 
- Under command k0 forecast for n1: p0 is identified as yellow and square; p3 is identified as green and cross; other positions are not identified. 
- Under command k1 forecast for n1: p1 is identified as blue and triangle; p3 is identified as green and cross; other positions are not identified. 
- Under command k2 forecast for n1: p2 is identified as red and circle; p3 is identified as green and cross; other positions are not identified. 
- Under command k3 forecast for n1: p3 is identified as green and cross; other positions are not identified.

Selected position at each buffer under each commanded forecast (from internal predicted-by-command records; where the record gives no selection forecast I note that no selection is provided): 
- Command k0 (forecast): n0 → p2; n1 → no selection forecast provided; n2 → p0. 
- Command k1 (forecast): n0 → p2; n1 → no selection forecast provided; n2 → p1. 
- Command k2 (forecast): n0 → p2; n1 → no selection forecast provided; n2 → p2. 
- Command k3 (forecast): n0 → p2; n1 → no selection forecast provided; n2 → p3.

All statements above are drawn from the record’s measured events and internal forecasts. Attributes shown as absent or omitted reflect that the record supplies no identified label or no selection forecast for that node/position.


## 1321_r2_c0_remapped

Summary of current selected positions (distinguishing forecasts and observations)

- q7: Forecasted selection (predicted_current) is p1. The most recent measured allocation in the history (last command k3) also placed q7 on p1 (acquisition quality 0.17157).  
- q9: Forecasted selection (predicted_current) is p3. The most recent measured allocation in history (last command k3) placed q9 on p3 (acquisition quality 0.96249).  
- q2 (output_node): There is no forecasted current selection in the record and no recent measured allocation for q2.

How access at each buffer position would change over two steps without refresh (predicted recovery at delay 0 and delay 2; trend)

- q7 (forecast):  
  - p0: recovery falls from about 0.67 to about 0.39; trend declining.  
  - p1: recovery falls from about 0.17 to about 0.10; trend declining.  
  - p2: recovery falls from about 0.29 to about 0.16; trend declining.  
  - p3: recovery falls from about 0.13 to about 0.10; trend declining.

- q9 (forecast):  
  - p0: recovery falls from about 0.33 to about 0.16; trend declining.  
  - p1: recovery falls from about 0.09 to about 0.06; trend declining.  
  - p2: recovery is about 0.033 at delay 0 and about 0.028 at delay 2; trend steady.  
  - p3: recovery falls from about 0.87 to about 0.56; trend declining.

- q2 (output_node): The record provides no recovery-by-delay forecasts for q2 positions, so there is no evidence about how access would change over two steps without refresh.

Identified output colors and shapes (output_node q2)

- Current forecast (predicted_current for q2):  
  - p3 is identified as green and cross.  
  - p0, p1 and p2 have no identified color or shape in the current forecast (identifiers are null).

- Forecasts under each command (predicted_by_command for q2):  
  - Command k0: p0 identified as yellow and square; p3 identified as green and cross. p1 and p2 are not identified.  
  - Command k1: p1 identified as blue and triangle; p3 identified as green and cross. p0 and p2 are not identified.  
  - Command k2: p2 identified as red and circle; p3 identified as green and cross. p0 and p1 are not identified.  
  - Command k3: p3 identified as green and cross. p0, p1 and p2 are not identified.

Selected position at each buffer under each command (predicted_by_command; these are forecasts)

- Command k0: q7 → p2; q9 → p0. q2 has no selection forecast under k0 in the record.  
- Command k1: q7 → p2; q9 → p1. q2 has no selection forecast under k1 in the record.  
- Command k2: q7 → p2; q9 → p2. q2 has no selection forecast under k2 in the record.  
- Command k3: q7 → p2; q9 → p3. q2 has no selection forecast under k3 in the record.

Notes: where the record shows null for an identified attribute I have reported that the position is not identified; where selection or recovery information is absent I have reported no evidence. All “current” statements labeled as forecasts come from the model’s predicted_current outputs; observed allocations cited above come from the measured observed_history entries.


## 1321_r2_c0_conflicting_description

Current selected positions (forecasts vs measured):
- n0: Predicted currently to select p1. Measured history contains many past allocations to n0 (observed one‑hot allocations to p0–p3 on different trials), but those are past events, not direct evidence that the present selection differs from the current forecast.
- n1 (output node): The record does not provide a current selected position readout; selection is omitted.
- n2: Predicted currently to select p3. Observed history shows repeated past allocations to p3 and p0 on different trials; those are measured past events.

How access at each buffer position would change over two steps without refresh (forecasts):
- n0: For every position p0, p1, p2 and p3 the model predicts a loss of access over the two-step delay (access is lower after two steps than at the start).
- n1: No simulated multi-step recovery data are present for n1; change over delays is not provided.
- n2: Positions p0, p1 and p3 are predicted to lose access over two steps. Position p2 is predicted to remain about the same across the two-step interval.

Identified colors and shapes in the output readout (n1), current and under each command (forecasts):
- Current forecast for n1: p3 is identified as green and cross. Other positions at n1 are not identified in the current forecast.
- If command k0 is executed (forecast): n1 would show p0 identified as yellow square and p3 as green cross; other positions at n1 would remain without identified attributes.
- If command k1 is executed (forecast): n1 would show p1 identified as blue triangle and p3 as green cross; other positions at n1 would remain without identified attributes.
- If command k2 is executed (forecast): n1 would show p2 identified as red circle and p3 as green cross; other positions at n1 would remain without identified attributes.
- If command k3 is executed (forecast): n1 would show p3 identified as green cross; other positions at n1 would remain without identified attributes.

Selected position at each buffer under each command (forecasts; where selection info is present):
- Command k0: n0 predicted to select p2; n1 selection is not provided; n2 predicted to select p0.
- Command k1: n0 predicted to select p2; n1 selection is not provided; n2 predicted to select p1.
- Command k2: n0 predicted to select p2; n1 selection is not provided; n2 predicted to select p2.
- Command k3: n0 predicted to select p2; n1 selection is not provided; n2 predicted to select p3.

Notes: I have reported measured allocation events separately from internal forecasts and have omitted any attribute for which the record gives no identifying value. The informal operator comment is not used as evidence.


## 1321_r2_c0_missing_relation

Summary note: there are no measured events in observed_history. All statements about current allocations and access are from the system's internal forecasts (predicted_current). There are no command forecasts to report.

Current selected position at each buffer (predicted, not observed)
- n0: predicted to be selecting position p1.
- n1 (the output readout): no selection allocation is present in the forecast; no dominant selected position is reported.
- n2: predicted to be selecting position p3.

How access at each buffer position would change over two steps without refresh (predicted)
- n0
  - p0: starts relatively high accessibility and would fall over the two-step interval.
  - p1: starts at a lower accessibility and would decline further over the two steps.
  - p2: begins at moderate accessibility and would decline over two steps.
  - p3: begins low and would decline further over two steps.
  All four positions at n0 show a declining trend across the two delays.
- n1
  - No recovery-by-delay forecasts are present for any p0–p3 at n1; there is no internal prediction of how access would change without refresh for this buffer.
- n2
  - p3: starts very high accessibility and would decrease over the two steps but remain the most accessible of its positions.
  - p0: begins at moderate accessibility and would decline.
  - p1: begins low and would decline.
  - p2: begins very low and shows essentially no change across the two steps (steady).

Identified colors and shapes (categorical reductions) — current and under commands
- n1 (output readout) currently identifies at position p3 the color green and the shape cross. All other positions at n1 have no identified color or shape in the forecast.
- There are no command forecasts, so there is no predicted change in the output-node identifications under any command.
- For completeness (forecasted in other buffers): n0 forecasts identify p0 as yellow and square; n2 forecasts also identify p3 as green and cross. These are internal categorical reductions reported by the buffers, not observed events.

Selected positions under commands
- There are no command forecasts; therefore no alternative selected-position predictions under commands are available.


## 1321_r2_c0_model_swap

Current selected position (forecasts): according to the internal forecasts, node n0 is predicted to select p1 and node n2 is predicted to select p3. The readout node n1 (the output node) has no selection forecast in the current prediction; no observed record in history supplies a current selected position for any node.

How access would change over two steps without refresh (forecasts): for n0 every position shows a falling simulated-access value over two delay steps; the forecast labels this trend as declining at p0, p1, p2 and p3 (access would be lower after two steps than now for all four positions). For n2 three positions (p0, p1 and p3) are also forecast to decline over two steps; p2 is forecast as steady across the two-step interval. For n1 there is no recovery-by-delay forecast in the record, so change over two steps is not provided.

Identified output colors and shapes (output node n1)
- Current (forecast): only position p3 at n1 has identified attributes: green and cross. Positions p0, p1 and p2 at n1 have no identified color or shape in the current forecast.
- Under each commanded forecast:
  - k0: n1 would include p2 identified as red and circle, and p3 identified as green and cross; other positions have no identified attributes in that forecast.
  - k1: n1 would include p2 identified as red and circle, and p3 identified as green and cross.
  - k2: n1 would include p2 identified as red and circle, and p3 identified as green and cross.
  - k3: n1 would include p2 identified as red and circle, and p3 identified as green and cross.

Selected positions under each command (forecasts): the predicted selected positions from the command forecasts are as follows.
- k0: n0 → p0; n1 → no selection forecast provided; n2 → p2.
- k1: n0 → p1; n1 → no selection forecast provided; n2 → p2.
- k2: n0 → p2; n1 → no selection forecast provided; n2 → p2.
- k3: n0 → p3; n1 → no selection forecast provided; n2 → p2.

Observed events (measured history): the recorded history contains measured allocation and acquisition events for n0 and n2 under past commands, but those measurements do not include explicit selected-position readouts and do not change the above forecasts; where a forecast is missing, the record provides no evidence of a current or commanded selection.


## 1321_r2_c0_restored

Current selected positions (based on internal forecasts, not direct observation)
- n0: predicted to be selecting p1 (this is a forecast from predicted_current).
- n1 (output_node): the record gives no current selection readout; no selected position is supported.
- n2: predicted to be selecting p3 (forecast from predicted_current).

Note on measured history: the observed history contains past allocation and acquisition events for n0 and n2 but does not show explicit selected positions, so those events do not establish a current selection.

How access at each buffer position would change over two steps without refresh (forecasts)
- n0 (predicted_current): every position shows a falling access pattern over two steps. p0 begins at the highest access and drops substantially; p2 starts at a middling level and drops to lower access; p1 and p3 start relatively low and fall further.
- n2 (predicted_current): p3 begins with the highest access and declines over two steps; p0 begins moderate and declines; p1 begins low and declines; p2 is very low and remains essentially unchanged across the two steps.
- n1 (output_node): the record does not include access-over-delay forecasts for positions at n1, so no evidence about two-step access change is available.

Identified colors and shapes at the output node (n1)
- Current forecast (predicted_current): p3 is identified as green and cross. Other n1 positions are not identified in the current forecast.
- Forecasts under each command (predicted_by_command; these are conditional forecasts, not observed outcomes):
  - Under k0: n1 p0 identified as yellow and square; n1 p3 identified as green and cross. Other n1 positions are not identified in that forecast.
  - Under k1: n1 p1 identified as blue and triangle; n1 p3 identified as green and cross. Other n1 positions are not identified.
  - Under k2: n1 p2 identified as red and circle; n1 p3 identified as green and cross. Other n1 positions are not identified.
  - Under k3: n1 p3 identified as green and cross; other n1 positions are not identified.

Selected position at each buffer under each command (forecasts from predicted_by_command)
- Command k0:
  - n0 predicted to select p2.
  - n1: no selection readout provided in the k0 forecast (no supported selected position).
  - n2 predicted to select p0.
- Command k1:
  - n0 predicted to select p2.
  - n1: no selection readout provided in the k1 forecast.
  - n2 predicted to select p1.
- Command k2:
  - n0 predicted to select p2.
  - n1: no selection readout provided in the k2 forecast.
  - n2 predicted to select p2.
- Command k3:
  - n0 predicted to select p2.
  - n1: no selection readout provided in the k3 forecast.
  - n2 predicted to select p3.

Summary of limits: selections and access-change descriptions above are taken from internal forecasts in the record. Where identified attributes are null or selection distributions are absent, the record provides no supported identification or selected position. Observed history supplies allocation and acquisition events but not current selections.


## 1321_r2_c1_neutral

Current selected positions (forecasts, from predicted_current):
- n0: p3.
- n2: p1.
- n1 (output node): no selection forecast present in the record.

How access would change over two steps without refresh (predicted_current recovery at delays 0→1→2 and unattended trend):
- n0:
  - p0: recovery falls from about 0.127 to 0.111 to 0.090; trend declining.
  - p1: recovery falls from about 0.059 to 0.049 to 0.040; trend steady.
  - p2: recovery falls from about 0.042 to 0.036 to 0.032; trend steady.
  - p3: recovery falls from about 0.236 to 0.194 to 0.159; trend declining.
- n2:
  - p0: recovery falls from about 0.613 to 0.457 to 0.345; trend declining.
  - p1: recovery falls from about 0.870 to 0.696 to 0.531; trend declining.
  - p2: recovery falls from about 0.130 to 0.102 to 0.077; trend declining.
  - p3: recovery falls from about 0.319 to 0.233 to 0.171; trend declining.
- n1 (output node): recovery_by_delay values are not provided in the predicted_current record, so no forecast of two-step access change is available there.

Identified output colors and shapes (output node n1)
- Current (predicted_current): p0 identified as yellow square; p1 identified as blue triangle; p2 and p3 have no identified color or shape in the record.
- Under commands (predicted_by_command for n1):
  - k0: p0 yellow square; p1 blue triangle; p2 red circle; p3 none identified.
  - k1: p0 yellow square; p1 blue triangle; p2 red circle; p3 none identified.
  - k2: p0 yellow square; p1 blue triangle; p2 red circle; p3 none identified.
  - k3: p0 has no identified color/shape; p1 blue triangle; p2 red circle; p3 none identified.

Selected positions under each command (predicted_by_command; forecasts):
- k0:
  - n0 → p0.
  - n1 → no selection forecast present.
  - n2 → p2.
- k1:
  - n0 → p1.
  - n1 → no selection forecast present.
  - n2 → p2.
- k2:
  - n0 → p2.
  - n1 → no selection forecast present.
  - n2 → p2.
- k3:
  - n0 → p3.
  - n1 → no selection forecast present.
  - n2 → p2.

Notes: all above about selected positions, recovery and identified attributes come from the internal forecasts in the record. Where fields are absent in the record I report that no forecast or recovery values are available.


## 1321_r2_c1_remapped

All statements below refer only to what the record contains. Forecasts from the model’s internal readout are labeled as forecasts; measured events come from observed_history.

Current selected position (forecasts unless noted)
- q7: forecast selects p3.
- q2 (the output node): no current selection forecast is provided in the record.
- q9: forecast selects p1.

How access at each buffer position would change over two steps without refresh
- q7: forecasted access would decline for p0 and p3 over two steps; it would remain essentially steady for p1 and p2.
- q2: no multi-step access forecasts are provided for any positions in the current readout.
- q9: forecasted access at every position declines over two steps.

Identified output colors and shapes (q2, the readout)
- Current forecast (internal): p0 is identified as yellow and square; p1 is identified as blue and triangle; p2 has no identified color or shape in the record; p3 has no identified color or shape in the record.
- If the system executed command k0 (forecast): q2 would identify p0 as yellow/square, p1 as blue/triangle, p2 as red/circle, and p3 would remain without identified color or shape.
- If k1 were executed (forecast): q2 would identify p0 as yellow/square, p1 as blue/triangle, p2 as red/circle, and p3 would remain without identified attributes.
- If k2 were executed (forecast): q2 would identify p0 as yellow/square, p1 as blue/triangle, p2 as red/circle, and p3 would remain without identified attributes.
- If k3 were executed (forecast): q2 would identify p1 as blue/triangle and p2 as red/circle; p0 and p3 would have no identified color or shape.

Selected position at each buffer under each command (forecasts from predicted_by_command)
- Command k0: q7 forecast selects p0; q2 selection not forecasted; q9 forecast selects p2.
- Command k1: q7 forecast selects p1; q2 selection not forecasted; q9 forecast selects p2.
- Command k2: q7 forecast selects p2; q2 selection not forecasted; q9 forecast selects p2.
- Command k3: q7 forecast selects p3; q2 selection not forecasted; q9 forecast selects p2.

Measured events (observed_history)
- The record contains multiple measured allocation and acquisition events for q7 and q9 across past commands; those are measurements, not the current-selection forecasts above.


## 1321_r2_c1_conflicting_description

Current selected positions (forecasts): n0 is predicted to select p3; n2 is predicted to select p1. There is no predicted current selection reported for n1 (the output readout), so no supported claim about n1’s selected position from the record.

How access would change over two steps without refresh (forecasts): 
- n0: access at p0 and p3 is predicted to decline over two steps; access at p1 and p2 is predicted to remain basically steady. These come from the buffer’s simulated recovery over delays. 
- n2: access at every position (p0–p3) is predicted to decline over two steps. 
- n1: the record provides no simulated recovery-by-delay for positions, so change over two steps is not reported.

Identified output colors and shapes at n1 (the output readout)
- Current (forecast): p0 — color: yellow, shape: square; p1 — color: blue, shape: triangle; p2 — color and shape not identified; p3 — color and shape not identified.
- If command k0 were executed (forecast): p0 — yellow / square; p1 — blue / triangle; p2 — red / circle; p3 — not identified.
- If command k1 were executed (forecast): p0 — yellow / square; p1 — blue / triangle; p2 — red / circle; p3 — not identified.
- If command k2 were executed (forecast): p0 — yellow / square; p1 — blue / triangle; p2 — red / circle; p3 — not identified.
- If command k3 were executed (forecast): p0 — color and shape not identified; p1 — blue / triangle; p2 — red / circle; p3 — not identified.

Selected positions under each command (forecasts):
- Command k0: n0 predicted to select p0; n2 predicted to select p2. No selection prediction is reported for n1.
- Command k1: n0 predicted to select p1; n2 predicted to select p2. No selection prediction is reported for n1.
- Command k2: n0 predicted to select p2; n2 predicted to select p2. No selection prediction is reported for n1.
- Command k3: n0 predicted to select p3; n2 predicted to select p2. No selection prediction is reported for n1.

Observed events versus forecasts and missing information: the allocation and acquisition entries in observed_history are measured past events; all the selected positions, recovery trends and identified attributes above are taken from the internal forecasts in predicted_current and predicted_by_command. Where an identified color or shape is null in the record I have called it not identified; where recovery or selection data are absent I have reported that no forecast is provided.


## 1321_r2_c1_missing_relation

There are no measured events in the record; everything below derives from internal forecasts.

Current selected positions (forecasts)
- n0 (buffer): the forecasted selection is at p3.
- n2 (buffer): the forecasted selection is at p1.
- n1 (the output readout): no selection distribution is reported, so there is no forecast of a selected position for n1.

How access at each buffer position would change over two steps without refresh (forecasts)
- n0 (buffer): access is forecast to fall at p0 and at p3 over the two-step interval; access at p1 and at p2 is forecast to remain essentially unchanged.
- n2 (buffer): access is forecast to fall at every position (p0, p1, p2, p3) over the two-step interval.
- n1: no recovery forecasts are provided for its positions, so the record gives no basis to state how access there would change.

Identified output colors and shapes (categorical reductions; forecasts, not observed)
- n1 (output readout): p0 is categorized as color yellow and shape square; p1 is categorized as color blue and shape triangle. p2 and p3 have no identified color or shape in the record (those attributes are omitted).
- n2 (buffer) also contains the same categorical reductions for p0 (yellow, square) and p1 (blue, triangle); p2 and p3 are unidentified there as well.
- n0 has no identified color or shape for any position in the forecast.

Selected positions under commands
- The record contains no command-conditioned forecasts, so there is no evidence about how selected positions at any node would change under any command.


## 1321_r2_c1_model_swap

Summary of what the record supports.

Current selected positions (forecasts): predicted_current forecasts that n0 is selecting p3 and n2 is selecting p1. predicted_current provides no selection readout for n1 (output_node), so there is no forecasted selected position there. The observed history lists past commands, allocations and acquisitions but does not directly show current selections; therefore the above are forecasted, not measured, selections.

How access at each buffer position would change over two steps without refresh (forecasts): For n0 the forecasted access for positions p0 and p3 would fall over two steps; access for p1 and p2 is forecasted to remain roughly steady. For n2 the forecasted access for every position (p0–p3) would fall over two steps. For n1 there is no recovery_by_delay information in the forecast, so the record gives no evidence about how access there would change without refresh.

Identified output (n1) colors and shapes now and under each command:
- Current (predicted_current, readout): n1 at p0 is identified as yellow and square; n1 at p1 is identified as blue and triangle; n1 at p2 and p3 have no identified color or shape in the current forecast.
- Under command k0 (predicted_by_command): n1 p0 = yellow / square; p1 = blue / triangle; p2 and p3 remain without identifications in that forecast.
- Under command k1 (predicted_by_command): n1 p0 = yellow / square; p1 = blue / triangle; p2 and p3 not identified.
- Under command k2 (predicted_by_command): n1 p0 = yellow / square; p1 = blue / triangle; p2 = red / circle; p3 not identified.
- Under command k3 (predicted_by_command): n1 p1 = blue / triangle and n1 p3 = green / cross; n1 p0 and p2 are not identified in that forecast.

Selected positions at each buffer under each command (forecasts from predicted_by_command):
- Command k0: n0 predicted to select p2; n2 predicted to select p0; n1 has no selection prediction in that forecast.
- Command k1: n0 predicted to select p2; n2 predicted to select p1; n1 has no selection prediction.
- Command k2: n0 predicted to select p2; n2 predicted to select p2; n1 has no selection prediction.
- Command k3: n0 predicted to select p2; n2 predicted to select p3; n1 has no selection prediction.

Notes on missing or unidentified information: where the record shows identified_color or identified_shape as null that means the model did not assign a dominant category for that position; where fields such as selection_distribution or recovery_by_delay are absent for a node, the record provides no evidence about selection or access changes for that node. Observed_history entries are measured past events but do not replace the internal forecasts used above.


## 1321_r2_c1_restored

Output node is n1.

Current selected positions (forecasts unless noted)
- n0: forecast selects p3.
- n1 (output): no selection forecast is recorded, so current selected position is not provided.
- n2: forecast selects p1.

How access would change over two steps without refresh (predicted unattended change)
- n0 positions: p0 — access is predicted to decline; p1 — predicted to remain roughly steady; p2 — predicted steady; p3 — predicted to decline.
- n1 positions: no simulated multi-step recovery predictions are provided for n1 in the record.
- n2 positions: all stored positions (p0, p1, p2, p3) are predicted to show declining access over two steps.

Identified output colors and shapes at n1 now (from the output readout)
- p0: color yellow, shape square.
- p1: color blue, shape triangle.
- p2: color not identified, shape not identified.
- p3: color not identified, shape not identified.

Identified output colors and shapes at n1 under each predicted command (forecasts)
- Command k0 (forecast): p0 yellow/square; p1 blue/triangle; p2 red/circle; p3 none identified.
- Command k1 (forecast): p0 yellow/square; p1 blue/triangle; p2 red/circle; p3 none identified.
- Command k2 (forecast): p0 yellow/square; p1 blue/triangle; p2 red/circle; p3 none identified.
- Command k3 (forecast): p0 not identified; p1 blue/triangle; p2 red/circle; p3 not identified.

Selected position at each buffer under each predicted command (forecasts)
- Command k0: n0 -> p0; n1 -> p0; n2 -> p2.
- Command k1: n0 -> p1; n1 -> p1; n2 -> p2.
- Command k2: n0 -> p2; n1 -> p2; n2 -> p2.
- Command k3: n0 -> p3; n1 -> no selection forecast recorded; n2 -> p2.

Notes
- The observed history contains measured allocation and acquisition events for n0 and n2 from prior commands, but the current selected positions and multi-step access changes reported above come from the internal forecasts in the record. Unidentified attributes are reported as such; omitted forecasts or recovery predictions are noted where absent.

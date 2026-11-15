# VFD-200 Frequency Converter — Service Manual (fictional sample)

This document is a fictional sample written for testing the assistant. The device, codes and values are invented. Do not use it for real equipment.

## Safety warnings

High voltage remains on the DC bus for up to 5 minutes after the supply is switched off. Before opening the cover, isolate the supply, apply lockout/tagout, wait 5 minutes and confirm with a multimeter that the DC bus voltage is below 50 V DC.

Only qualified electricians may work inside the converter.

## Technical data

Supply voltage: 3 × 400 V AC, 50/60 Hz. Rated output current: 18 A. Output frequency range: 0–120 Hz. Ambient temperature: −10 °C to +45 °C; above 40 °C derate output current by 2 % per °C. Enclosure: IP54. Cooling: internal fan, replace every 30 000 operating hours.

## Alarm codes

F-0011 Overvoltage on DC bus. Cause: deceleration time too short or supply voltage too high. Action: increase deceleration ramp (parameter P-305) or fit a braking resistor.

F-0021 Overcurrent at output. Cause: short circuit in motor cable, motor winding fault or acceleration too fast. Action: disconnect the motor, measure insulation resistance of the motor and cable (minimum 1 MΩ at 500 V DC), then increase acceleration ramp (parameter P-304).

F-0034 Heatsink overtemperature (above 90 °C). Cause: blocked air inlet, failed cooling fan or ambient temperature above 45 °C. Action: clean the air filter, check that the cooling fan runs, check cabinet ventilation.

F-0050 Earth fault. Cause: insulation breakdown between an output phase and earth. Action: do not reset repeatedly; measure insulation of motor and cable before restarting.

W-0102 Fieldbus communication lost (warning). Cause: Modbus RTU master stopped polling for longer than the timeout set in P-602 (default 2 s). Action: check the RS-485 cable, termination resistor (120 Ω at both ends) and the master's polling cycle.

## Parameters

P-304 Acceleration time, 0.1–600 s, default 10 s.
P-305 Deceleration time, 0.1–600 s, default 10 s.
P-601 Modbus slave address, 1–247, default 1.
P-602 Fieldbus timeout, 0.5–60 s, default 2 s.
P-900 Parameter lock: 0 = parameters editable, 1 = read-only. Set to 1 after commissioning so settings cannot be changed from the keypad without the service code.

## Resetting an alarm

Press STOP/RESET for 2 seconds after the cause has been removed. Earth fault F-0050 can only be reset after a full power cycle. Write down the alarm code and the operating hours before resetting.

## Maintenance schedule

Every 6 months: clean air filter, check terminal tightness (torque 2.5 Nm for power terminals), check fan operation. Every 5 years or 30 000 hours: replace cooling fan. Every 10 years: replace DC bus capacitors.

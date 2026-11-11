"""Compare a probe's NAME/hex output to independently evaluated C macros."""
import json, pathlib, sys
reference_path = pathlib.Path(sys.argv[1])
expected = (json.loads(reference_path.read_text()) if reference_path.suffix == '.json' else
            dict((name, int(value)) for name, value in
                 (line.split() for line in reference_path.read_text().splitlines())))
skip = set(json.loads(pathlib.Path(__file__).with_name('unavailable-constants.json').read_text()))
actual = dict((name, int(value, 16)) for name, value in
              (line.split() for line in pathlib.Path(sys.argv[2]).read_text().splitlines()))
failures = []
for name in sorted(expected.keys() - skip):
    value = actual.get(name)
    reference = expected[name]
    # C and Rust may widen a signed 32-bit mask differently, preserving its bits.
    equal = value == reference or (value is not None and reference <= 0xffffffff
             and value >= 0xffffffff80000000 and value & 0xffffffff == reference)
    if not equal:
        failures.append((name, reference, value))
for failure in failures:
    print('MISMATCH', *failure)
print(f'libc constants: {len(expected.keys() - skip)} checked, {len(failures)} failures')
sys.exit(bool(failures))

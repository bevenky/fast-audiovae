"""Check path expansion and flags without compiling, loading or running native code."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('amd_build_adapter', ROOT / 'build_aocl.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
recipe = json.loads((ROOT / 'pins/rebuild.json').read_text())
roots = {key: Path('/supplied inputs') / key for key in
         ('RELEASE', 'AOCL_CORE_SOURCE', 'AOCL_ROOT', 'LIBXSMM', 'ORT_INCLUDE', 'NATIVE_BUILD', 'OBJECTS', 'OUTPUT')}
native = Path('/supplied inputs/alternate native.so')
commands = adapter.compile_commands(recipe, roots, native, 'custom-cc', 'custom-cxx')
assert len(commands) == 8
for index, (original, actual) in enumerate(zip(recipe['object_compile_commands'] + recipe['aocl_link_commands'], commands)):
    assert len(original) == len(actual)
    assert actual[0] == ('custom-cc' if index < 2 else 'custom-cxx')
    assert all('${' not in token for token in actual)
    for before, after in zip(original[1:], actual[1:]):
        if '${' not in before:
            assert before == after
assert all('-c' in command and '-ffp-contract=off' in command for command in commands[:4])
assert all(str(native) in command for command in commands[-2:])
assert '-Wl,-rpath,$ORIGIN' in commands[5]
assert all(not any('mkl' in item.lower() for item in command) for command in commands)
assert '-DFX_WITH_LIBXSMM=1' in commands[0] and '-DUP_WITH_LIBXSMM=1' in commands[1]
assert '-DIP_WITH_AOCL=1' in commands[4]
try:
    adapter.expand('${UNRESOLVED}/x', roots)
except ValueError:
    pass
else:
    raise AssertionError('Unknown placeholder accepted')
print(json.dumps({'status': 'passed', 'commands_checked': 8, 'native_execution': False, 'compilation': False}))

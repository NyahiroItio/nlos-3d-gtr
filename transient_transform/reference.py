"""Prepare a MATLAB reference run without changing the upstream source file.

The unmodified author entry point is run first. A separate instrumented copy
only captures intermediate arrays and adds two small deterministic fixtures;
all original reconstruction statements are retained unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .upstream import DEFAULT_ROOT, SOURCE_SHA256, verify_source


DEFAULT_OUTPUT = Path(__file__).resolve().parent/'results'/'original_reference'
DEFAULT_FIXTURE = Path(__file__).resolve().parent/'tests'/'fixtures'/'author_matlab_reference.npz'


def matlab_path(path):
    return str(Path(path).resolve()).replace('\\', '/').replace("'", "''")


def prepare(source_dir=DEFAULT_ROOT/'cnlos2018', output=DEFAULT_OUTPUT):
    source = verify_source(source_dir).read_text(encoding='utf-8')
    output = Path(output).resolve()
    generated = output/'instrumentation'
    generated.mkdir(parents=True, exist_ok=True)
    # Modern MATLAB resolves the implicitly loaded variable `width` as its
    # newer built-in function unless it is explicitly declared in this scope.
    # The load immediately overwrites []; no numerical statement is changed.
    compat = source.replace('function cnlos_reconstruction(scene)',
                            'function cnlos_2018_compat(scene)', 1)
    compat = compat.replace('% Default scene', 'width = []; % R2024b load/name compatibility\n% Default scene', 1)
    compat_path = generated/'cnlos_2018_compat.m'
    compat_path.write_text(compat, encoding='utf-8')
    changes = []

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'Upstream instrumentation anchor changed: {old}')
        source = source.replace(old, new, 1)
        changes.append(old)

    replace('function cnlos_reconstruction(scene)',
            'function result = cnlos_reference_probe(scene, output_dir)')
    replace('% Default scene', 'width = []; % R2024b load/name compatibility\n% Default scene')
    replace('switch scene', '''switch scene
    case {0,-1}
        sample = 1:(4*4*64);
        rect_data = reshape(2+sin(sample./7)+0.1*cos(sample./11),[4 4 64]);
        width = 0.01;
        z_offset = 0;
        z_trim = 0;
        if scene == -1
            isdiffuse = 1;
            snr = snr.*1e-1;
        end''')
    replace('data = permute(rect_data,[3 2 1]);',
            'data = permute(rect_data,[3 2 1]);\nprepared_data = data;')
    replace('% Step 2: Resample time axis and pad result',
            'weighted_data = data;\n% Step 2: Resample time axis and pad result')
    replace('% Step 3: Convolve with inverse filter and unpad result',
            'resampled_data = tdata(1:M,1:N,1:N);\n% Step 3: Convolve with inverse filter and unpad result')
    replace('% Step 4: Resample depth axis and clamp results',
            'filtered_data = tvol;\n% Step 4: Resample depth axis and clamp results')
    replace('vol  = max(real(vol),0);',
            'signed_volume = real(vol);\nvol = max(real(vol),0);\nclipped_volume = vol;')
    capture = f'''result = struct();
result.source_sha256 = '{SOURCE_SHA256}';
result.prepared_data = prepared_data;
result.weighted_data = weighted_data;
result.resampled_data = resampled_data;
result.filtered_data = filtered_data;
result.signed_volume = signed_volume;
result.clipped_volume = clipped_volume;
result.display_volume = vol;
result.mtx = full(mtx);
result.mtxi = full(mtxi);
result.tic_x = tic_x;
result.tic_y = tic_y;
result.tic_z = tic_z;
result.width = width;
result.range = range;
result.bin_resolution = bin_resolution;
result.snr = snr;
result.isdiffuse = isdiffuse;
result.z_offset = z_offset;
result.z_trim = z_trim;
result.N = N;
result.M = M;
if M <= 32
    result.psf = psf;
    result.inverse_filter = invpsf;
end
save(fullfile(output_dir,sprintf('scene_%d.mat',scene)),'-struct','result','-v7');
% View result'''
    replace('% View result', capture)
    probe = generated/'cnlos_reference_probe.m'
    probe.write_text(source, encoding='utf-8')
    driver = output/'run_reference.m'
    driver.write_text(f"""set(0,'DefaultFigureVisible','off');
cd('{matlab_path(source_dir)}');
addpath('{matlab_path(generated)}');
fprintf('Running unchanged author entry point, scene 7...\\n');
try
    cnlos_reconstruction(7);
    raw_status = struct('succeeded',true,'message','Unchanged original entry point ran.');
catch exception
    raw_status = struct('succeeded',false,'identifier',exception.identifier,'message',getReport(exception));
    if ~contains(raw_status.message,'width')
        rethrow(exception);
    end
end
handle = fopen('{matlab_path(output/'unmodified_entry_status.json')}','w');
fprintf(handle,'%s',jsonencode(raw_status));
fclose(handle);
close all;
fprintf('Running original arithmetic with width declaration compatibility fix...\\n');
cnlos_2018_compat(7);
exportgraphics(gcf,'{matlab_path(output/'author_compat_scene7.png')}','Resolution',200);
close all;
for scene = [0 -1 7 9]
    fprintf('Capturing original arithmetic, scene %d...\\n',scene);
    result = cnlos_reference_probe(scene,'{matlab_path(output)}');
    if scene > 0
        exportgraphics(gcf,fullfile('{matlab_path(output)}',sprintf('matlab_scene%d.png',scene)),'Resolution',200);
    end
    clear result;
    close all;
end
fprintf('ORIGINAL_LCT_REFERENCE_COMPLETE\\n');
""", encoding='utf-8')
    (output/'provenance.json').write_text(json.dumps({
        'source_sha256': SOURCE_SHA256,
        'probe_sha256': hashlib.sha256(probe.read_bytes()).hexdigest(),
        'compatibility_source_sha256': hashlib.sha256(compat_path.read_bytes()).hexdigest(),
        'compatibility_change': 'Declare width=[] before load to avoid R2024b built-in name resolution; loaded width overwrites it.',
        'instrumentation_anchors': changes,
        'scope': 'Original file untouched. Compatibility declaration, captures and two synthetic inputs; no original arithmetic replaced.',
    }, indent=2), encoding='utf-8')
    return driver


def export_fixtures(reference_dir=DEFAULT_OUTPUT, output=DEFAULT_FIXTURE):
    """Freeze small deterministic MATLAB results for tests without MATLAB/data."""
    import numpy as np
    from scipy.io import loadmat
    arrays = {'source_sha256': np.array(SOURCE_SHA256)}
    fields = ('prepared_data', 'weighted_data', 'resampled_data', 'filtered_data',
              'signed_volume', 'clipped_volume', 'display_volume', 'mtx', 'mtxi',
              'psf', 'inverse_filter', 'tic_x', 'tic_y', 'tic_z', 'width', 'range',
              'snr', 'isdiffuse', 'z_offset', 'z_trim', 'N', 'M', 'bin_resolution')
    for scene, label in ((0, 'retro'), (-1, 'diffuse')):
        reference = loadmat(Path(reference_dir)/f'scene_{scene}.mat', squeeze_me=True)
        if str(reference['source_sha256']) != SOURCE_SHA256:
            raise ValueError('Fixture reference does not identify the pinned author source.')
        for name in fields:
            arrays[f'{label}_{name}'] = np.asarray(reference[name])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_ROOT/'cnlos2018')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--export-fixtures', action='store_true')
    parser.add_argument('--fixture-output', type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    print(export_fixtures(args.output, args.fixture_output) if args.export_fixtures
          else prepare(args.source_dir, args.output))


if __name__ == '__main__':
    main()

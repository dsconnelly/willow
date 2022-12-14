import os
import shutil

from typing import Optional

from ..utils.copying import copy_with_modifications

def initialize_coupled_run(
    base_dir: str,
    model_dir: str,
    n_years: int=15,
) -> None:
    """
    Set up a MiMA run coupled with a data-driven emulator for online testing.

    Parameters
    ----------
    base_dir : Directory containing the MiMA run from which to copy the batch
        submission script and namelist, with appropriate modifications, as well
        as the model executable.
    model_dir : Directory containing the trained emulator.
    n_years : Number of years to integrate MiMA for.

    """

    base_dir = os.path.abspath(base_dir)
    model_dir = os.path.abspath(model_dir)

    model_name = os.path.basename(model_dir)
    case_dir = os.path.join(os.path.dirname(base_dir), model_name)
    os.makedirs(os.path.join(case_dir, 'RESTART'), exist_ok=True)

    src = os.path.join(base_dir, 'INPUT')
    dst = os.path.join(case_dir, 'INPUT')
    shutil.copytree(src, dst, dirs_exist_ok=True)

    for fname in ['field_table', 'mima.x']:
        src = os.path.join(base_dir, fname)
        shutil.copy2(src, case_dir)

    src = os.path.join(base_dir, 'diag_table')
    copy_with_modifications(src, case_dir, _diag_table_modifier)

    src = os.path.join(base_dir, 'input.nml')
    args: tuple = (os.path.join(model_dir, 'model.pkl'),)
    copy_with_modifications(src, case_dir, _namelist_modifier, *args)

    src = os.path.join(base_dir, 'submit.slurm')
    args = (base_dir, case_dir, model_name, n_years)
    copy_with_modifications(src, case_dir, _submit_modifier, *args)

def _diag_table_modifier(line: str) -> Optional[str]:
    """
    Modifier function for `diag_table`.

    Parameters
    ----------
    line : Line from the original `diag_table`.

    Returns
    -------
    line : Same as `line`, unless `line` specified output of specific humidity,
        buoyancy frequency, or precipitation, in which case `None` is returned.

    """

    drop = ['sphum', 'bf_cgwd', 'precip']
    if any([s in line for s in drop]):
        return None

    return line

def _namelist_modifier(line: str, model_path: str) -> str:
    """
    Modifier function for `input.nml`.

    Parameters
    ----------
    line : Line from the original `input.nml`.
    model_path : Path where trained emulator is saved.

    Returns
    -------
    line : Same as `line`, unless `line` specified whether `forpy` should be
        used, in which case lines pointing to the pickled emulator are returned.

    """

    if 'use_forpy' in line:
        lines = [
            'use_forpy = .true.,',
            f'forpy_model_path = \'{model_path}\' /'
        ]

        return ''.join([f'    {s}\n' for s in lines])

    return line

def _submit_modifier(
    line: str,
    base_dir: str,
    case_dir: str,
    model_name: str,
    n_years: int
) -> str:
    """
    Modifier function for `submit.slurm`.

    Parameters
    ----------
    line : Line from the original `submit.slurm`.
    base_dir : Directory containing the original MiMA run.
    case_dir : Directory where the new MiMA run is being set up.
    model_name : Name of the emulator being coupled.
    n_years : Number of years to integrate MiMA for.

    Returns
    -------
    line : Same as `line`, with potential modifications to do with slurm job
        parameters, data management, and integration time.

    """

    if '--out' in line:
        return line.replace(base_dir, case_dir)

    if '--mem' in line:
        memory = '16G'
        if 'random' in model_name or 'mubofo' in model_name:
            memory = '64G'

        return line.replace('8G', memory)

    if '--job-name' in line:
        return f'#SBATCH --job-name={model_name}-online\n'

    if 'openmpi' in line:
        return line + 'module load cdo/intel/1.9.10\n'

    if line.startswith('cd'):
        src = os.path.join(base_dir, 'INPUT', '*')
        dst = os.path.join(case_dir, 'INPUT')

        return line.replace(base_dir, case_dir) + f'cp {src} {dst}\n'

    if './mima.x' in line:
        return 'export PYTHONWARNINGS=ignore::UserWarning\n' + line

    if '{01..40}' in line:
        return line.replace('{01..40}', f'{{01..{n_years:02}}}')

    if 'mv RESTART/* INPUT' in line:
        lines = [
            line.strip(), '',
            'cdo --reduce_dim \\',
            '    -daymean -zonmean \\',
            '    -selname,u_gwf \\',
            '    ${yy}/atmos_4xdaily.nc ${yy}/tmp.nc', '',
            'cdo --reduce_dim -mermean \\',
            '    -sellonlatbox,0.0,360.0,-5.0,5.0 \\',
            '    ${yy}/tmp.nc ${yy}/qbo.nc', '',
            'cdo -mermean \\',
            '    -sellonlatbox,0.0,360.0,58.0,60.5 \\',
            '    ${yy}/tmp.nc ${yy}/ssw.nc', '',
            'rm ${yy}/atmos_4xdaily.nc',
            'rm ${yy}/tmp.nc'
        ]

        return ''.join([f'    {s}\n' if s else '\n' for s in lines])

    if 'rm .model.run' in line:
        lines = [
            line.strip(), '',
            'cdo mergetime ??/qbo.nc qbo.nc',
            'cdo mergetime ??/ssw.nc ssw.nc',
            f'for yy in {{01..{n_years:02}}}; do',
            '    rm -rf ${yy}',
            'done'
        ]

        return ''.join([s + '\n' for s in lines])

    return line

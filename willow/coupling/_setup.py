import os
import shutil

def setup_mima(control_dir, model_dir, case_dir=None):
    """
    Set up a MiMA run with an emulator for online testing.

    Parameters
    ----------
    control_dir : str
        The directory of the control run used to train the model.
    model_dir : str
        The directory where the trained model is saved.
    case_dir : str
        The directory where the new MiMA run should be created. If None, uses
        the parent directory of control_dir with the base name of model_dir.

    """
    
    model_name = os.path.basename(model_dir)
    if case_dir is None:
        parent_dir = os.path.dirname(control_dir)
        case_dir = os.path.join(parent_dir, model_name)
    
    os.makedirs(os.path.join(case_dir, 'RESTART'), exist_ok=True)
    shutil.copytree(
        os.path.join(control_dir, 'INPUT'), 
        os.path.join(case_dir, 'INPUT'),
        dirs_exist_ok=True
    )
    
    fnames = ['diag_table', 'field_table', 'mima.x']
    for fname in fnames:
        shutil.copy2(os.path.join(control_dir, fname), case_dir)
        
    model_path = os.path.abspath(os.path.join(model_dir, 'model.pkl'))
    _modify_copy(
        os.path.join(control_dir, 'input.nml'),
        os.path.join(case_dir, 'input.nml'),
        _input_modifier,
        model_path
    )
    
    _modify_copy(
        os.path.join(control_dir, 'submit.slurm'),
        os.path.join(case_dir, 'submit.slurm'),
        _submit_modifier,
        control_dir, case_dir, model_name
    )
          
def _get_lines(path):
    with open(path) as f:
        return f.readlines()
        
def _input_modifier(line, model_path):
    if 'use_forpy' in line:
        return  (
            '     use_forpy = .true.,\n'
            f'     forpy_model_path = \'{model_path}\' /\n'
        )
    
    return line

def _modify_copy(src, dst, modifier, *args):
    with open(dst, 'w') as f:
        for line in _get_lines(src):
            f.write(modifier(line, *args))

def _submit_modifier(line, control_dir, case_dir, model_name):
    if control_dir in line:
        return line.replace(control_dir, case_dir)

    if '--mem' in line:
        return line.replace('8G', '16G')
    
    if 'job-name' in line:
        return f'#SBATCH --job-name={model_name}\n'

    if './mima.x' in line:
        return 'export PYTHONWARNINGS=ignore::UserWarning\n' + line

    if '{01..40}' in line:
        return line.replace('{01..40}', '{01..05}')
    
    return line
    

    
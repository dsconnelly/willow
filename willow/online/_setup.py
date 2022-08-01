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
    
    if case_dir is None:
        parent_dir = os.path.dirname(control_dir)
        model_name = os.path.basename(model_dir)
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
    with open(os.path.join(case_dir, 'input.nml'), 'w') as f:
        for line in _get_lines(os.path.join(control_dir, 'input.nml')):
            if 'use_forpy' in line:
                f.write('     use_forpy = .true.,\n')
                f.write(f'     forpy_model_path = \'{model_path}\' /\n')
                
            else:
                f.write(line)
                
    with open(os.path.join(case_dir, 'submit.slurm'), 'w') as f:
        for line in _get_lines(os.path.join(control_dir, 'submit.slurm')):
            if control_dir in line:
                line = line.replace(control_dir, case_dir)
                
            f.write(line)
    
def _get_lines(path):
    with open(path) as f:
        return f.readlines()
    
        
    
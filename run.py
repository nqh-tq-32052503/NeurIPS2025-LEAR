import nbformat
import subprocess
def prepare_notebook(notebook_path):
    # Đọc file notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Ép thông tin kernel chuẩn của Kaggle vào metadata của notebook
    nb.metadata['kernelspec'] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    nb.metadata['language_info'] = {
        "codemirror_mode": { "name": "ipython", "version": 3 },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.12" 
    }

    # Ghi đè lại file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

folder = "./run"
target_nb = folder + "/template.ipynb"
prepare_notebook(target_nb)
subprocess.run(["kaggle", "kernels", "push", "-p", folder])
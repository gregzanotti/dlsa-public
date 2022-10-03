#1/bin/bash

# start jupyter server
#!/bin/bash

nohup jupyter-lab --allow-root --notebook-dir=/mnt/ssd0 --ip=* --port=8888 --no-browser --NotebookApp.token='' > /dev/null 2>&1 &
# CMD ["jupyter-lab", "--allow-root", "--notebook-dir=/mnt/ssd0", "--ip=*", "--port=8888", "--no-browser", "&"]

echo "JupyterLab server started."

conda activate dlsa

echo "DLAP environment activated."


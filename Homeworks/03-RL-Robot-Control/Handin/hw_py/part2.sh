# export JAX_TRACEBACK_FILTERING=off
export MUJOCO_GL=egl
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'cuda' | paste -sd: -)
export LD_LIBRARY_PATH=''
python part2_getup.py

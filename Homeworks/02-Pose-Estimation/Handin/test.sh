# Task1
python test.py \
    --checkpoint=exps/est_pose_final/checkpoint/exp_pose.pth \
    --mode=val \
    --device=cuda

# Task2
python test.py \
    --checkpoint=exps/est_coord_final/checkpoint/exp_coord.pth \
    --mode=val \
    --device=cuda

# Task 3
python eval.py \
    --checkpoint=exps/est_pose_final/checkpoint/exp_pose.pth \
    --mode=val \
    --device=cuda \
    --vis=0 \
    --headless=1

python eval.py \
    --checkpoint=exps/est_coord_final/checkpoint/exp_coord.pth \
    --mode=val \
    --device=cuda \
    --vis=0 \
    --headless=1

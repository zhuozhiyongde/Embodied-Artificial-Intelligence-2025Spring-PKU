# Task 1
python train.py \
    --model_type=est_pose \
    --exp_name=est_pose_final \
    --device=cuda \
    --max_iter=20000 \
    --batch_size=32

# Task 2
python train.py \
    --model_type=est_coord \
    --exp_name=est_coord_final \
    --device=cuda \
    --max_iter=20000 \
    --batch_size=32
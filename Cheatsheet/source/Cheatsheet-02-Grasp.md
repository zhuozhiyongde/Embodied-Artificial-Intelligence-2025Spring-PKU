# Vision-and-Grasping

**抓握式操作** （Prehensile Manipulation）：通过完全约束物体自由度实现精确控制。**非抓握式操作**：利用推、滑等接触力学原理调整物体状态，适用于薄片状物体或预处理场景，**不是所有动作都需要抓取**

**抓取姿势（Grasp Pose）**：手的位置、方向和关节状态 & **抓取的自由度**

-   **4-DoF 抓取**：仅需平移和绕重力轴旋转 $(x, y, z, \theta_z)$
-   **6-DoF 抓取**：允许任意方向接近 $(x, y, z, \theta_x, \theta_y, \theta_z)$
-   **手指自由度**：平行夹爪：开 / 关，1 DoF；灵巧手（Dexterous Hand）：21 DoF

**开环抓取系统**：一般处理流程包括视觉感知、位姿估计、运动规划、控制执行，**闭环系统**：接收反馈信号重新抓取

**对已知物体的抓取条件**：预测物体位姿，RGB 图像需相机内参、物体大小（避免歧义 ambiguity）、 **无对称性**；点云只需 **无对称性**

**ICP**：初始化变换估计 $T_0 = (R_0, t_0)$，迭代（数据关联找变换后最近匹配点 + 变换求解 $T_{k+1} = \arg \min_T \sum_{(m,s) \in C} \| Tm - s \|^2$）直至收敛

**对未知物体的抓取方法**：直接预测抓取位姿，也有算法可以从见过同一类别物体进行泛化。

**旋转回归**：用神经网络估计连续的旋转变量，理想表达方式：表示空间到 $\mathbb{SO}(3)$ 群一一映射、且满足连续性，即不存在 **奇点（Singularities）**

**欧拉角**、**轴角**：存在多义性。**四元数**：双重覆盖问题，约束空间为上半球则引入不连续性（临近球大圆的不连续性和球大圆上的不连续性）。$q = \left[\cos\frac{\theta}{2}, \sin\frac{\theta}{2} \hat{\omega}\right]$，本质是 $\mathbb{R}^4$ 与 $\mathbb{SO}(3)$ 无法构成拓扑同构, 至少 $\mathbb{R}^5$ 才行

**6D 表示**：拟合旋转矩阵然后施密特正交化（Schmidt orthogonalization），但拟合的 9 个数不等价

**9D 表示**：SVD 正交化，CNN 友好，连续、一一映射、等价。$\hat{R} = U\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(UV) \end{bmatrix}V^{\top}$（保证 $\det(\hat{R}) = 1$，符合手性不变性）

**增量旋转预测 Delta Rotation**：小范围旋转，以上几种方式都适用，此时四元数等表示方法需要预测参数较少，学起来更快

**Rotation Fitting**：通过神经网络预测拟合物体表面点的对应关系（相机坐标系 $(x_i^C, y_i^C, z_i^C)$ 到模型坐标系  $(x_i^M, y_i^M, z_i^M)$ 的最优变换矩阵）。步骤：对物体表面的每个像素，预测其在物体建模模型上的 3D 坐标；基于这些对应关系拟合旋转矩阵（要求物体见过）

**Orthogonal Procrustes**：给定两组点 $\mathbf{M}, \mathbf{N} \in \mathbb{R}^{n \times 3}$，求最优旋转矩阵 $\mathbf{A}$ 使得 $\hat{\mathbf{A}} = \arg\min_{\mathbf{A}} \|\mathbf{M} - \mathbf{N}\mathbf{A}^\top\|_F^2, \quad \text{s.t.}~\mathbf{A}^\top\mathbf{A} = \mathbf{I}$，其中 $\|X\|_F = \sqrt{\text{trace}(X^{\top}X)} = \sqrt{\sum_{i,j} x_{ij}^2}$

解析解：对 $\mathbf{M}^\top\mathbf{N}$ 做 SVD 分解，$\mathbf{M}^\top\mathbf{N} = \mathbf{UDV}^\top$，最优旋转为 $\hat{\mathbf{A}} = \mathbf{U}\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(\mathbf{UV}^{\top}) \end{bmatrix}\mathbf{V}^{\top}$，$\hat{\mathbf{t}} = \overline{\mathbf{M}^{\top} - \hat{\mathbf{A}} \mathbf{N}^{\top}}$

问题：对离群点敏感，常用 RANSAC 进行鲁棒拟合。

**RANSAC**：通过随机抽样找到内点 (inliers) 最多的模型假设，1. 以最小点集采样拟合所需模型（线=2，面=3，旋转=3）；2. 计算模型参数；3. 计算内点数量（距离小于阈值 $\epsilon$ 的）；4. 迭代重复，找内点最多的模型。

**Instance Level Pose Estimation**：每个物体有独立模型，如 PoseCNN，结合 ICP 可提升精度

**Category Level Pose Estimation**：同类物体归一化到 $1\times1\times1$ box，可预测旋转，预测平移需已知物体大小

**迭代最近点算法（ICP）**：作为后处理提高物体的位姿估计精度，提高抓取成功率；平移误差比旋转误差更敏感；怕物体被挡住造成 **点云缺失**。**目标**：优化初始位姿估计，对齐源点云 $P = \{p_i\}$ 和目标点云 $Q = \{q_j\}$ ；寻找最优旋转 $\hat{R} \in\mathbb{SO}(3)$ 和平移 $\hat{T}\in\mathbb{R}^{3\times 1}$

流程：1. 中心化源、目标点集 $\tilde{p}_i = p_i - \bar{P}, \tilde{q}_j = q_j - \bar{Q}$；2. **对应点匹配（Correspondence Search）**：为每个 $\tilde{p}_i$ 找到最近邻 $\tilde{q}_{j_i}$；3. 求解位姿（解见前）；4. 用解变换源点集 $P$，然后迭代 2-3 直至收敛。

**ICP 收敛性**：**不保证全局收敛**，可能陷入局部最优。原因：**对应点匹配可能非一一映射**，两个源点映到同一目标点。**优点**：简单，无需特征提取，初始估计好时精度高。**缺点**：计算成本高（最近邻搜索），对初始位姿敏感，迭代慢，未充分利用结构信息。

**Category-Level Pose Estimation**：解决实例级位姿估计需完整模型的问题，**通过归一化操作定义标准化物体空间 Normalized Object Coordinate Space（NOCS）** ，包括旋转对齐、平移归一、尺寸归一。

**ICP 算法需要很强的先验知识（物体的本身建模）**，然后进行变换前后点云配准，由于需要变换前后的**坐标对**，所以我们需要先进行**最近邻匹配**（也就是这一步导致了收敛性的缺失以及迭代速度的变慢），然后据此迭代得到物体位姿 $(R,t)$

**NOCS 算法不需要完整的物体的本身建模**，而是通过标准化的 NOCS 空间隐式地引入了对于某一类物体的、相较于 ICP 算法**更粗粒度**的几何先验，降低了对于高精建模的依赖，使用**合成数据**训练得到一个神经网络，可以从 RGB 图像直接为每一个像素预测其在 NOCS 中的对应点 $(x,y,z)$，随后将其与 RGBD 重建得到的点云信息进行配准，**这里根据像素关系，可以天然形成数量相同的变换前后的坐标对，所以不再需要找到最近邻（Correspondence）**。而后，我们可以直接用 Umeyama 算法（和 ICP 去除最近邻匹配的后半段类似）来重建得到 7 DoF 物体位姿 $(s,R,t)$

1. 输入 RGBD 图像，提取 RGB 信息，使用 Mask R-CNN 获得 ROI（Region of Interest），分割物体
2. 预测每个像素的 NOCS 空间坐标 $(x,y,z)$，得到 **NOCS Map**
3. 将 NOCS Map 的点反投影（Back Projection）到三维空间中，得到点云数据 $\mathbf{q}_i$
4. 通过 NOCS Map 和 Depth 图像得到的点云数据，进行 Pose Fitting，利用 Umeyama 算法，计算得出物体的 7DoF 位姿（缩放 + 旋转 + 平移），缩放系数的计算就是简单的用 NOCS Map 的各轴向长度与物体实际点云各轴向作了一个除法。而反过来计算 Bounding Box 的时候，则利用了 NOCS 建模时令物体中心处在原点从而具有的对称性，以预测出的 NOCS Map 各轴向最大绝对值乘 2 再乘缩放系数作为了 Bounding Box 的各轴向尺寸

**额外引入 NCOS 而不是直接 NN 预测原始点然后结合 Depth 直接回归 6DoF 位姿的原因**：

1. 实验效果优
2. 将问题分解为 2D $\to$ 3D 映射 + 3D $\to$ 3D 几何优化，更直观
3. **NOCS 方法充分利用了形状 / 几何先验**，提升了对未见物体的泛化能力。

**Synthetic data 合成数据**：训练 NOCS 网络需要大量标注数据，但真实数据标注成本高、泛化性差，所以需要合成数据进行训练。然而存在 **Sim2Real Gap**，导致模型在真实世界性能下降

**Mixed Reality Data** ：将合成前景物体叠加到真实背景上，易于获取大量带 NOCS 标签的数据。问题：合成前景与背景 **分界太过明显**，从而导致分割的 Mask R-CNN 学习到的经验难以应用到真实世界

**Co-Training** ：结合图像分割领域收集的真实数据集与合成数据集来一同对 Mask R-CNN 进行 **混合训练**，但前者不参与后续的 NOCS 映射训练，**只为分割提供监督信号**

**后续处理**：对于预测得到的位姿，有时候还需要 Refinement，比如之前介绍的 ICP 算法，也可通过神经网络（合成数据训练）完成

**Form Closure**：纯几何约束，不依赖摩擦力，最稳固

**Force Closure**：依赖摩擦力，通过接触力抵抗任意 Wrench（力 + 力矩），也即可以让物体产生的任何加速度 $a$ 和角加速度 $\alpha$

$\text{Form Closure} \subseteq \text{Force Closure} \subseteq \text{Successful Grasp}$，反例：1. 双指夹纸 2. 托起

**摩擦锥（Friction Cone）**：定义了在静摩擦系数 $\mu$ 下，接触点不滑动的力的方向范围（与法线夹角最大值 $\alpha = \arctan \mu$）

**Force Closure 形式化**：定义抓取矩阵（Grasp Matrix）$F = \begin{bmatrix} \mathcal{F}_1 & \cdots & \mathcal{F}_j \end{bmatrix} \in \mathbb{R}^{n \times j},\ n = 3 \text{ or } 6,\ j = k \times C$

其中，$C$ 是接触点（摩擦锥）的数量，$k$ 是为了近似每个摩擦锥所使用的力旋量数量（也即用多少面体锥来近似摩擦锥）。

1. 抓取矩阵必须满秩： $\text{rank}(F)=n$
2. 原点在凸锥内部（小正数下限保证稳定性）：$Fk = 0 \text{ for some } k \in \mathbb{R}^j, k_i \ge \epsilon > 0 \text{ for all } i$ 

**GraspNet-1B**：获得物体模型 $\to$ 在表面采样抓取位姿 $\to$ Force Closure 筛选 $\to$ 场景生成与物体位姿标注 $\to$ 变化抓取位姿到场景中，进行碰撞检测 $\to$ 多视角渲染扩增数据集。**意义**：证明了基于 3D 几何的模型能有效学习抓取，但大规模多样性需要合成数据。

**摩擦系数 $\mu$**：GraspNet 数据集实际上为从低到高不同的 $\mu$ 值都进行了筛选并存储了标签。$\mu$ 值越低，对抓取的要求越高（更接近 Form Closure），这样的抓取在低摩擦表面上更可能成功。训练时可用小 $\mu$，泛化性强。

**Grasp Detection**：从输入（点云 / TSDF / 图像）预测抓取位姿，包括位置、朝向、夹爪宽度、质量分数。

**输入模态**：往往 3D 几何通常优于 2D 信息，**因其直接包含形状信息**。

**VGN（Volumetric Grasping Network）** 输入：3D TSDF；架构：U-Net（3D CNN）；输出：预测三个体素网格，即**对于输出网格中的每一个体素**，网络预测 **抓取质量（Grasp Quality/Score）** 、 **抓取朝向（Grasp Orientation）** 、 **抓取宽度（Grasp Width）** （防止夹爪过宽向外碰撞）

特点：**依赖几何信息**、不依赖纹理、Sim2Real 效果好（优势在于 TSDF 对传感器噪声不敏感；现实物理效应（力闭合、变形）可能优于仿真；可以工程优化摩擦系数问题），**Sim2Real 甚至可以是负的！**

评估指标：通常是与物体无关、非任务导向的 **抓取成功率（Success Rate）** 、 **清理率（Percentage Cleard）** 、 **规划时间（Planning Time）**

后处理：**高斯平滑**（抓取评分非阶跃）、 **距离掩膜**（限制夹爪在 TSDF 的区域） 、 **基于质量的 NMS**

**VGN 的局限性**：依赖多视角构建完整的场景 TSDF、精度受体素大小限制、高分辨率计算所需的内存成本高

**GraspNet 成功的本质：**

1. 点云的优良性质：准确、轻量、效率高、精度高
2. 架构：**端到端网络**，**多阶段设计**（先预测抓取分数，选高的做候选继续预测接近方向等），每个阶段都有监督信号，稳定
3. **泛化性**：局部性与平移等变性
    1. **局部性**：Cylinder Grouping 聚合，依赖候选点周围的局部几何信息判断，而不太关心场景中其他远处的物体。
    2. **平移等变性（Translation Equivariance）**：类似二维情形，模型学习到的几何模式识别能力不随物体在空间中的位置变化而失效。

**Cylinder Grouping**：在候选点附近，沿着接近方向定义一个圆柱体区域，聚合该区域内所有点的特征。这个聚合后的特征被用来预测最佳的旋转角和深度 / 宽度。

GraspNet 的核心在于 **学习局部几何特征（Local Geometric Features）与抓取成功的关系**，泛化性强

**抓取的条件生成模型**：替代检测方法，直接学习抓取位姿的分布（解决多峰分布）；尤其适用于高自由度灵巧手；常用条件扩散模型，基于局部几何特征生成抓取。

**DexGraspNet**：合成数据（Synthetic Data） + 深度学习

1. 场景理解：预测每个点 **抓取可能性（Graspness）**，是否是 **物体（Objectness）**
2. 局部特征：不用全局特征（关联性弱、泛化性差），选择 Graspness 高的地方附近的点云，提取局部特征（**几何信息**）
3. 条件抓取生成模块：**条件生成处理 $(T, R)$ 多峰分布**，然后采样后直接预测手指形态 $\theta$

问题：仅处理包覆式抓取（Power Grasp），没处理指尖抓取（Precision Grasp）；主要使用力封闭抓取；透明（Transparent）或高反光（Highly Specular/Shiny）物体有折射（Refraction）/ 镜面反射（Specular Reflection），导致点云质量差。

**ASGrasp**：深度修复，合成数据 + 监督学习。**域随机化**、多模态立体视觉、立体匹配（Stereo Matching） 。

**Affordance 可供性**：指一个物体所能支持或提供的交互方式或操作可能性，哪个区域、何种方式进行交互。

**Where2Act**：大量随机尝试 + 标注。学习从视觉输入预测交互点 $a_p$、交互方向 $R_{z|p}$ 和成功置信度 $s_{R|p}$。**VAT-Mart**：预测一整条操作轨迹。

利用视觉输入进行预测：

-   **物体位姿（Object Pose）**：需要模型、抓取标注。
-   **抓取位姿（Grasp Pose）**：直接预测抓取点和姿态，无模型或预定义抓取。
-   **可供性（Affordance）**

**启发式（Heuristic）规则**：预抓取 Pre-grasp，到附近安全位置再闭合，避免碰撞

1.  **操作复杂度有限**：难以处理复杂任务，受启发式规则设计限制。
2.  **开环执行（Open-loop）**：规划一次，执行到底，闭眼做事。高频重复规划可近似闭环。

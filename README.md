逻辑回归广泛应用于CTR估计，本实现采用Per-Coordinate FTRL-Proximal算法来训练LR模型。
此算法是online learning的，所需内存很小，可以流式训练。目前只实现了单机版(后续版本将实现基于Downpour SGD 的多机版)。
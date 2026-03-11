from yacs.config import CfgNode as CN

class Config(object):
    r"""
    Configuration class for the model, training, and testing.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    is_train: bool
        Whether the configuration is for training or testing.
    """

    def __init__(self, config_yaml: str, is_train=True):
        # Initialize the root configuration node
        self._C = CN()

        # MODEL configuration
        self._C.MODEL = CN()
        self._C.MODEL.image_model_name = 'net_image'
        self._C.MODEL.video_model_name = 'net_video1'
        self._C.MODEL.n_feats = 16

        if is_train:
            # TRAINING configuration
            self._C.TRAINING = CN()
            self._C.TRAINING.gpu_ids = [0, 1]
            self._C.TRAINING.batch_size = 8
            self._C.TRAINING.num_workers = 32
            self._C.TRAINING.num_epochs = 100
            self._C.TRAINING.lr = 2e-4
            self._C.TRAINING.grad_clip = 0.1
            self._C.TRAINING.resume_model = False
            self._C.TRAINING.checkpoint_path = './checkpoints'
            self._C.TRAINING.num_frames = 5

            # DATASETS configuration for training and validation
            self._C.DATASETS = CN()
            self._C.DATASETS.train = CN()
            self._C.DATASETS.train.raw_root = '/path-to-train-dataset/input'
            self._C.DATASETS.train.gt_root = '/path-to-train-dataset/gt'
            self._C.DATASETS.train.size = [640, 480]
            self._C.DATASETS.train.use_shuffle = True

            self._C.DATASETS.val = CN()
            self._C.DATASETS.val.raw_root = '/path-to-val-dataset/input'
            self._C.DATASETS.val.gt_root = '/path-to-train-dataset/gt'
            self._C.DATASETS.val.size = [640, 480]
            self._C.DATASETS.val.use_shuffle = False
        else:
            # TESTING configuration
            self._C.TESTING = CN()
            self._C.TESTING.gpu_ids = [0]
            self._C.TESTING.checkpoint_path = './checkpoints'
            self._C.TESTING.result_path = './results/'
            self._C.TESTING.num_frames = 5

            # DATASETS configuration for testing
            self._C.DATASETS = CN()
            self._C.DATASETS.test = CN()
            self._C.DATASETS.test.dataset_name = 'UVEB'
            self._C.DATASETS.test.raw_root = '/path-to-test-dataset/input'
            self._C.DATASETS.test.size = [640, 480]
            self._C.DATASETS.test.use_shuffle = False

        # Override parameter values from YAML file
        if config_yaml:
            self._C.merge_from_file(config_yaml)

        # Make the configuration immutable
        self._C.freeze()

    def dump(self, file_path: str):
        r"""
        Save the configuration to a YAML file.

        Parameters
        ----------
        file_path: str
            Path to save the configuration YAML file.
        """
        with open(file_path, "w") as f:
            self._C.dump(stream=f)

    def __getattr__(self, attr: str):
        return getattr(self._C, attr)

    def __repr__(self):
        return repr(self._C)
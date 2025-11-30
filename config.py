import os

class Config:
    """
    Central configuration for dataset, models, and attacks.
    Designed to work in Google Colab inside /content/InfoSec-Work.
    """

    # -------------------------
    # DATASET PATHS
    # -------------------------
    DATA_ROOT = r"H:/University sht/INFO-SEC/research/content/InfoSec-Work/img-data/data/"

    TRAIN_DIR = None
    TEST_DIR = None
    VAL_DIR = None

    @staticmethod
    def update_paths():
        """Updates dataset paths based on DATA_ROOT."""
        Config.TRAIN_DIR = os.path.join(Config.DATA_ROOT, "train")
        Config.TEST_DIR = os.path.join(Config.DATA_ROOT, "test")
        Config.VAL_DIR = os.path.join(Config.DATA_ROOT, "val")

    # -------------------------
    # TRAINING SETTINGS
    # -------------------------
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_EPOCHS = 15
    LR = 1e-4

    # -------------------------
    # MODEL SAVE PATHS
    # -------------------------
    SAVE_DIR = r"H:/University sht/INFO-SEC/research/content/InfoSec-Work/models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    XCEPTION_PATH = os.path.join(SAVE_DIR, "xception.pth")
    RESNET_PATH = os.path.join(SAVE_DIR, "resnet50.pth")
    MESONET_PATH = os.path.join(SAVE_DIR, "mesonet.pth")

    # -------------------------
    # ATTACK SETTINGS
    # -------------------------
    EPS_LIST = [1/255, 2/255, 4/255, 8/255]
    PGD_STEPS = 40
    PGD_ALPHA = 1/255

    # -------------------------
    # IMAGE SETTINGS
    # -------------------------
    IMG_SIZE = 299
        

# ✅ CALL UPDATE_PATHS AFTER CLASS IS CREATED
Config.update_paths()
